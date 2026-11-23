using AIManagement.Api.Repositories;
using OpenCvSharp;

namespace AIManagement.Api.Services;

public enum SnapshotMode
{
    EveryFace,      // image upload: save every face once
    BestPerSession  // live stream: one snapshot per person per visit, improved in place
}

public sealed class SnapshotService
{
    // ---- Tunable settings ----
    private const int MinFaceSize = 30;
    private const float MinConfidence = 0.3f;
    private const float TargetConfidence = 0.80f;   // "good enough": lock the snapshot
    private const float MinImprovement = 0.02f;     // a new frame must beat the saved one by this much
    private const float SameUnknownThreshold = 0.35f;
    private static readonly TimeSpan SessionGap = TimeSpan.FromMinutes(5);
    private static readonly TimeSpan MaxSessionAge = TimeSpan.FromHours(12);

    // Crop margins, as a fraction of the face box
    private const double PadSide = 0.6;
    private const double PadTop = 0.9;    // includes hair / turban
    private const double PadBottom = 0.5;

    private const string Unknown = "UNKNOWN";

    private sealed class Session
    {
        public string EmployeeId = "";
        public float[] Embedding = Array.Empty<float>();
        public float Confidence;
        public DateTime LastSeen;
        public DateTime Created;
        public int SightingId;
        public string Path = "";
        public bool Locked;
    }

    private readonly IServiceScopeFactory _scopes;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly List<Session> _sessions = new();

    public string Root { get; }

    public SnapshotService(IWebHostEnvironment environment, IServiceScopeFactory scopes)
    {
        _scopes = scopes;
        Root = Path.Combine(environment.ContentRootPath, "Snapshots");
        Directory.CreateDirectory(Root);
    }

    public async Task OfferAsync(
        byte[] imageBytes, int x, int y, int width, int height,
        float confidence, float[] embedding,
        string employeeId, string name, float similarity,
        string source, SnapshotMode mode)
    {
        if (width < MinFaceSize || height < MinFaceSize || confidence < MinConfidence)
            return;

        await _gate.WaitAsync();
        try
        {
            if (mode == SnapshotMode.EveryFace)
            {
                var path = WriteCrop(imageBytes, x, y, width, height, employeeId);
                if (path != null)
                    await InsertAsync(employeeId, name, source, similarity, path);
                return;
            }

            // ---- BestPerSession ----
            var now = DateTime.UtcNow;
            _sessions.RemoveAll(s =>
                now - s.LastSeen > SessionGap || now - s.Created > MaxSessionAge);

            var session = FindSession(employeeId, embedding);

            if (session == null)
            {
                var path = WriteCrop(imageBytes, x, y, width, height, employeeId);
                if (path == null) return;

                var id = await InsertAsync(employeeId, name, source, similarity, path);

                _sessions.Add(new Session
                {
                    EmployeeId = employeeId,
                    Embedding = embedding,
                    Confidence = confidence,
                    LastSeen = now,
                    Created = now,
                    SightingId = id,
                    Path = path,
                    Locked = confidence >= TargetConfidence
                });
                return;
            }

            session.LastSeen = now;

            // Already good enough, or not clearly better than what is saved
            if (session.Locked || confidence < session.Confidence + MinImprovement)
                return;

            var newPath = WriteCrop(imageBytes, x, y, width, height, employeeId);
            if (newPath == null) return;

            var oldPath = session.Path;

            if (!await UpdateAsync(session.SightingId, similarity, newPath))
                session.SightingId = await InsertAsync(employeeId, name, source, similarity, newPath);

            DeleteFile(oldPath);

            session.Path = newPath;
            session.Embedding = embedding;
            session.Confidence = confidence;
            session.Locked = confidence >= TargetConfidence;
        }
        finally
        {
            _gate.Release();
        }
    }

    private Session? FindSession(string employeeId, float[] embedding)
    {
        if (employeeId != Unknown)
            return _sessions.FirstOrDefault(s => s.EmployeeId == employeeId);

        // Unknown people have no ID, so match by face similarity
        return _sessions
            .Where(s => s.EmployeeId == Unknown)
            .Select(s => (Session: s, Similarity: Cosine(s.Embedding, embedding)))
            .Where(t => t.Similarity > SameUnknownThreshold)
            .OrderByDescending(t => t.Similarity)
            .Select(t => t.Session)
            .FirstOrDefault();
    }

    private string? WriteCrop(byte[] imageBytes, int x, int y, int width, int height, string employeeId)
    {
        var safeId = new string(employeeId
            .Where(c => char.IsLetterOrDigit(c) || c is '-' or '_').ToArray());
        if (safeId.Length == 0) return null;

        using var image = Cv2.ImDecode(imageBytes, ImreadModes.Color);
        if (image.Empty()) return null;

        var x1 = Math.Max(0, x - (int)(width * PadSide));
        var y1 = Math.Max(0, y - (int)(height * PadTop));
        var x2 = Math.Min(image.Width, x + width + (int)(width * PadSide));
        var y2 = Math.Min(image.Height, y + height + (int)(height * PadBottom));
        if (x2 <= x1 || y2 <= y1) return null;

        using var crop = new Mat(image, new Rect(x1, y1, x2 - x1, y2 - y1));

        var folder = DateTime.Now.ToString("yyyy-MM-dd");
        var fileName = $"{safeId}_{DateTime.Now:HHmmss}_{Guid.NewGuid().ToString("N")[..6]}.jpg";
        Directory.CreateDirectory(System.IO.Path.Combine(Root, folder));

        Cv2.ImWrite(System.IO.Path.Combine(Root, folder, fileName), crop);

        return $"{folder}/{fileName}";
    }

    private async Task<int> InsertAsync(string employeeId, string name, string source, float similarity, string path)
    {
        using var scope = _scopes.CreateScope();
        var repository = scope.ServiceProvider.GetRequiredService<FaceRepository>();
        return await repository.AddSightingAsync(employeeId, name, source, similarity, path);
    }

    private async Task<bool> UpdateAsync(int id, float similarity, string path)
    {
        using var scope = _scopes.CreateScope();
        var repository = scope.ServiceProvider.GetRequiredService<FaceRepository>();
        return await repository.UpdateSightingAsync(id, similarity, path);
    }

    private void DeleteFile(string relativePath)
    {
        var full = Resolve(relativePath);
        if (full != null && File.Exists(full)) File.Delete(full);
    }

    private static float Cosine(float[] a, float[] b)
    {
        if (a.Length == 0 || a.Length != b.Length) return 0;
        float dot = 0, na = 0, nb = 0;
        for (var i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return dot / (MathF.Sqrt(na) * MathF.Sqrt(nb) + 1e-9f);
    }

    // Turns a stored relative path into a full path, refusing anything outside Root.
    public string? Resolve(string relativePath)
    {
        var root = System.IO.Path.GetFullPath(Root);
        var full = System.IO.Path.GetFullPath(System.IO.Path.Combine(root, relativePath));
        return full.StartsWith(root + System.IO.Path.DirectorySeparatorChar) ? full : null;
    }
}