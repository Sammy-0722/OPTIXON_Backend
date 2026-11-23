using System.Collections.Concurrent;
using System.Text.Json.Serialization;
using AIManagement.Api.Repositories;
using AIManagement.Api.Services;
using OpenCvSharp;

namespace AIManagement.Api.services;

public sealed record PersonRow(string EmployeeId, string Name, float Similarity,
    DateTime FirstSeen, DateTime LastSeen, bool Present);

public sealed record AlertRow(DateTime At, string? SnapshotUrl);

public sealed class CameraState
{
    public string CameraId { get; init; } = "";
    public string Status { get; set; } = "starting";   // starting | online | offline | error
    public string? Error { get; set; }
    public DateTime? LastFrameAt { get; set; }
    public int PhonesInHand { get; set; }
    public bool AlertActive { get; set; }
    public List<AlertRow> Alerts { get; set; } = new();
    public List<PersonRow> People { get; set; } = new();
    public int BestVersion { get; set; }
    [JsonIgnore] public CameraOverlay? Overlay { get; set; }
    [JsonIgnore] public byte[]? BestJpeg { get; set; }
    public double HeldSeconds { get; set; }
}

public sealed class CameraMonitor : BackgroundService
{
    private const int FrameIntervalMs = 500;       // ~2 analysed frames per second per camera
    private const double AlertAfterSeconds = 20;   // same rule as the browser version
    private const double HoldSeconds = 2;          // keeps a phone "held" through short misses
    private const double ClearAfterSeconds = 3;    // phone gone this long = reset the timer
    private const float FaceSimilarityThreshold = 0.40f;   // same as RecognizeStream

    // One face inference at a time across all cameras (DirectML)
    private static readonly SemaphoreSlim FaceGate = new(1, 1);

    private readonly CameraStore _store;
    private readonly StreamHub _hub;
    private readonly PhoneUseService _phoneUse;
    private readonly PhoneDetectionService _phones;
    private readonly PhoneSnapshotService _snapshots;
    private readonly FaceInferenceService _faceInference;
    private readonly FaceRepository _faceRepo;
    private readonly StickyIdentityService _sticky;
    private readonly SnapshotService _faceSnapshots;
    private readonly string _baseUrl;

    private readonly ConcurrentDictionary<string, (string Signature, CancellationTokenSource Cts)> _running = new();
    public ConcurrentDictionary<string, CameraState> States { get; } = new();

    public CameraMonitor(
        CameraStore store, StreamHub hub, PhoneUseService phoneUse, PhoneDetectionService phones,
        PhoneSnapshotService snapshots, FaceInferenceService faceInference, FaceRepository faceRepo,
        StickyIdentityService sticky, SnapshotService faceSnapshots, IConfiguration config)
    {
        _store = store; _hub = hub; _phoneUse = phoneUse; _phones = phones; _snapshots = snapshots;
        _faceInference = faceInference; _faceRepo = faceRepo; _sticky = sticky; _faceSnapshots = faceSnapshots;
        _baseUrl = config["PublicBaseUrl"] ?? "https://localhost:7112";
    }

    // Every 5 seconds: start workers for new cameras, restart changed ones, stop removed ones
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        while (!stoppingToken.IsCancellationRequested)
        {
            try { Reconcile(stoppingToken); }
            catch (Exception ex) { Console.WriteLine($"[monitor] reconcile failed: {ex.Message}"); }

            try { await Task.Delay(5000, stoppingToken); }
            catch (OperationCanceledException) { break; }
        }

        foreach (var r in _running.Values) r.Cts.Cancel();
    }

    private void Reconcile(CancellationToken stoppingToken)
    {
        var wanted = _store.All().Where(c => c.Modules.Count > 0).ToDictionary(c => c.Id);

        foreach (var id in _running.Keys.Where(id => !wanted.ContainsKey(id)).ToList())
            Stop(id);

        foreach (var cam in wanted.Values)
        {
            var signature = cam.Url + "|" + string.Join(",", cam.Modules.OrderBy(m => m));
            if (_running.TryGetValue(cam.Id, out var current))
            {
                if (current.Signature == signature) continue;
                Stop(cam.Id);                          // url or modules changed: restart
            }

            var cts = CancellationTokenSource.CreateLinkedTokenSource(stoppingToken);
            _running[cam.Id] = (signature, cts);
            _ = Task.Run(() => RunCamera(cam, cts.Token));
        }
    }

    private void Stop(string id)
    {
        if (_running.TryRemove(id, out var r)) r.Cts.Cancel();
        States.TryRemove(id, out _);
    }

    private static byte[] Paint(Mat frame, CameraOverlay overlay)
    {
        using var canvas = frame.Clone();
        OverlayPainter.Draw(canvas, overlay);
        return canvas.ToBytes(".jpg");
    }

    private async Task<List<FaceInferenceResult>> AnalyzeFaces(byte[] jpeg, CancellationToken ct)
    {
        await FaceGate.WaitAsync(ct);
        try { return (await _faceInference.AnalyzeFacesAsync(jpeg, ct)).ToList(); }
        finally { FaceGate.Release(); }
    }

    private async Task RunCamera(CameraRecord cam, CancellationToken ct)
    {
        var state = States.GetOrAdd(cam.Id, id => new CameraState { CameraId = id });
        var doPhone = cam.Modules.Contains("phone");
        var doFace = cam.Modules.Contains("face");
        var streamKey = $"cam:{cam.Id}";

        var recentHeld = new List<(PhoneDetection Box, DateTime Time)>();
        DateTime? heldSince = null;
        var lastHeld = DateTime.MinValue;
        var alerted = false;

        var people = new Dictionary<string, PersonRow>();
        var bestScore = 0f;
        var lastAnything = DateTime.MinValue;

        while (!ct.IsCancellationRequested)
        {
            try
            {
                var (jpeg, error) = await _hub.GetFrameAsync(cam.Url);
                if (jpeg == null)
                {
                    state.Status = "offline"; state.Error = error; state.Overlay = null;
                    await Task.Delay(5000, ct);
                    continue;
                }

                using var frame = Cv2.ImDecode(jpeg, ImreadModes.Color);
                if (frame.Empty()) { await Task.Delay(FrameIntervalMs, ct); continue; }

                var now = DateTime.UtcNow;
                state.Status = "online"; state.Error = null; state.LastFrameAt = now;

                var faceBoxes = new List<OverlayBox>();
                var phoneBoxes = new List<OverlayBox>();
                var score = 0f;
                var fireAlert = false;

                // ---------------- FACE ----------------
                if (doFace)
                {
                    var faces = await AnalyzeFaces(jpeg, ct);

                    var input = new List<StickyFace>();
                    foreach (var f in faces)
                    {
                        var match = await _faceRepo.FindMatchAsync(f.Embedding);
                        var ok = match.HasValue && match.Value.Similarity >= FaceSimilarityThreshold;
                        input.Add(new StickyFace(
                            f.X, f.Y, f.Width, f.Height, f.Embedding, ok,
                            ok ? match.Value.EmployeeId : "UNKNOWN",
                            ok ? match.Value.Name : "Unknown",
                            match.HasValue ? match.Value.Similarity : 0f));
                    }

                    // always call it, even with no faces, so faces that left are released
                    var results = _sticky.Resolve(streamKey, input);

                    for (var i = 0; i < faces.Count; i++)
                    {
                        var f = faces[i];
                        var r = results[i];

                        faceBoxes.Add(new OverlayBox(f.X, f.Y, f.Width, f.Height,
                            r.Recognized ? $"{r.Name} {r.Similarity * 100:0}%" : "Unknown", r.Recognized));

                        score = Math.Max(score, r.Recognized ? f.DetectionConfidence : f.DetectionConfidence * 0.5f);

                        people[r.EmployeeId] = people.TryGetValue(r.EmployeeId, out var p)
                            ? p with { Name = r.Name, Similarity = Math.Max(p.Similarity, r.Similarity), LastSeen = now }
                            : new PersonRow(r.EmployeeId, r.Name, r.Similarity, now, now, true);

                        // saves the best crop per person; the raw frame has no boxes drawn on it
                        await _faceSnapshots.OfferAsync(
                            jpeg, f.X, f.Y, f.Width, f.Height, f.DetectionConfidence, f.Embedding,
                            r.EmployeeId, r.Name, r.Similarity, $"camera:{cam.Name}", SnapshotMode.BestPerSession);
                    }

                    state.People = people.Values
                        .OrderByDescending(p => p.LastSeen)
                        .Select(p => p with { Present = (now - p.LastSeen).TotalSeconds < 4 })
                        .ToList();
                }

                // ---------------- PHONE ----------------
                if (doPhone)
                {
                    var analysis = _phoneUse.Analyze(frame);

                    recentHeld.RemoveAll(h => (now - h.Time).TotalSeconds > HoldSeconds);
                    var held = analysis.Phones
                        .Where(u => u.Status != PhoneStatus.NotInHand ||
                                    recentHeld.Any(h => PhoneDetectionService.IoU(h.Box, u.Phone) > 0.3f))
                        .Select(u => u.Phone)
                        .ToList();
                    foreach (var p in held) recentHeld.Add((p, now));

                    state.PhonesInHand = held.Count;
                    foreach (var p in held)
                    {
                        phoneBoxes.Add(new OverlayBox(p.X, p.Y, p.Width, p.Height,
                            $"phone in hand {p.Confidence * 100:0}%", true));
                        score = Math.Max(score, 1f + p.Confidence);   // a phone beats any face for "best"
                    }

                    if (held.Count > 0)
                    {
                        lastHeld = now;
                        heldSince ??= now;
                        if (!alerted && (now - heldSince.Value).TotalSeconds >= AlertAfterSeconds)
                        {
                            alerted = true;
                            fireAlert = true;
                        }
                    }
                    else if ((now - lastHeld).TotalSeconds > ClearAfterSeconds)
                    {
                        heldSince = null; alerted = false; state.AlertActive = false;
                    }
                    state.HeldSeconds = held.Count > 0 && heldSince.HasValue ? (now - heldSince.Value).TotalSeconds : 0;
                }


                // ---------------- publish for the viewer ----------------
                var overlay = new CameraOverlay(faceBoxes, phoneBoxes, now);
                state.Overlay = faceBoxes.Count + phoneBoxes.Count > 0 ? overlay : null;

                if (score > 0) lastAnything = now;
                else if ((now - lastAnything).TotalSeconds > 10) bestScore = 0;

                if (score > bestScore + 0.02f)
                {
                    bestScore = score;
                    state.BestJpeg = Paint(frame, overlay);
                    state.BestVersion++;
                }

                if (fireAlert)
                {
                    state.AlertActive = true;
                    var saved = await _snapshots.SaveAsync(Paint(frame, overlay), null, _baseUrl, ct);
                    state.Alerts = state.Alerts.Prepend(new AlertRow(now, saved?.Url)).Take(20).ToList();
                    Console.WriteLine($"[monitor] ALERT {cam.Name}: phone in hand for {AlertAfterSeconds:0}s");
                }

                await Task.Delay(FrameIntervalMs, ct);
            }
            catch (OperationCanceledException) { break; }
            catch (Exception ex)
            {
                state.Status = "error"; state.Error = ex.Message;
                try { await Task.Delay(5000, ct); } catch (OperationCanceledException) { break; }
            }
        }
    }
}