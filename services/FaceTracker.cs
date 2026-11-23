namespace AIManagement.Api.services;

public sealed record FaceObservation(int X, int Y, int Width, int Height, float[] Embedding);

public sealed record SnapshotCandidate(
    byte[] Frame, int X, int Y, int Width, int Height,
    float Confidence, float[] Embedding,
    string EmployeeId, string Name, float Similarity, float Quality, int TrackId);

public sealed class IdentityVote
{
    public string Name = "";
    public int Count;
    public float SimilaritySum;
    public float BestSimilarity;
}

// One track = one person's appearance over time. Track identity (TrackId) is
// separate from the recognized person (Identity), which is decided by votes.
public sealed class TrackedFace
{
    public TrackedFace(int trackId, FaceObservation first, double now)
    {
        TrackId = trackId;
        Embedding = (float[])first.Embedding.Clone();
        LastBox = first;
        LastSeen = now;
        Frames = 1;
    }

    public int TrackId { get; }
    public float[] Embedding { get; private set; }
    public FaceObservation LastBox { get; private set; }
    public double LastSeen { get; private set; }
    public int Frames { get; private set; }

    public Dictionary<string, IdentityVote> Votes { get; } = new();

    // Best snapshot candidate seen so far for this track
    public SnapshotCandidate? Best { get; private set; }

    internal void Touch(FaceObservation observation, double now)
    {
        LastBox = observation;
        LastSeen = now;
        Frames++;

        // Follow slow appearance changes (pose, light)
        for (var i = 0; i < Embedding.Length && i < observation.Embedding.Length; i++)
            Embedding[i] = 0.8f * Embedding[i] + 0.2f * observation.Embedding[i];
    }

    // Call only for frames where this face matched a registered person
    public void Vote(string employeeId, string name, float similarity)
    {
        if (!Votes.TryGetValue(employeeId, out var vote))
            Votes[employeeId] = vote = new IdentityVote { Name = name };

        vote.Count++;
        vote.SimilaritySum += similarity;
        vote.BestSimilarity = Math.Max(vote.BestSimilarity, similarity);
    }

    // Keeps only the best-quality view of this track
    public void Offer(
        float quality, byte[] frame, FaceObservation box,
        float confidence, float similarity)
    {
        if (Best != null && quality <= Best.Quality) return;

        Best = new SnapshotCandidate(
            frame, box.X, box.Y, box.Width, box.Height,
            confidence, box.Embedding, "", "", similarity, quality, TrackId);
    }

    // The registered person this track most likely is, or null
    public (string EmployeeId, string Name)? Identity
    {
        get
        {
            if (Votes.Count == 0) return null;
            var top = Votes.OrderByDescending(v => v.Value.SimilaritySum).First();
            return (top.Key, top.Value.Name);
        }
    }
}

public sealed class FaceTracker
{
    private const double MatchSimilarity = 0.45;          // same face by appearance alone
    private const double MatchSimilarityWithOverlap = 0.20; // when the boxes also overlap
    private const double MinOverlap = 0.30;

    private readonly double _maxGapSeconds;
    private readonly List<TrackedFace> _active = new();
    private readonly List<TrackedFace> _finished = new();
    private int _nextId = 1;

    public FaceTracker(double maxGapSeconds)
    {
        _maxGapSeconds = maxGapSeconds;
    }

    // Links every face of one frame to a track. Returns one track per detection, in order.
    public TrackedFace[] Associate(IReadOnlyList<FaceObservation> detections, double now)
    {
        foreach (var old in _active.Where(t => now - t.LastSeen > _maxGapSeconds).ToList())
        {
            _active.Remove(old);
            _finished.Add(old);
        }

        var result = new TrackedFace?[detections.Count];
        var pairs = new List<(int Detection, TrackedFace Track, double Score)>();

        for (var i = 0; i < detections.Count; i++)
        {
            foreach (var track in _active)
            {
                var similarity = Cosine(track.Embedding, detections[i].Embedding);
                var overlap = IoU(track.LastBox, detections[i]);

                if (similarity >= MatchSimilarity ||
                    (overlap >= MinOverlap && similarity >= MatchSimilarityWithOverlap))
                {
                    pairs.Add((i, track, similarity + overlap));
                }
            }
        }

        var taken = new HashSet<TrackedFace>();

        foreach (var pair in pairs.OrderByDescending(p => p.Score))
        {
            if (result[pair.Detection] != null || taken.Contains(pair.Track)) continue;

            result[pair.Detection] = pair.Track;
            taken.Add(pair.Track);
            pair.Track.Touch(detections[pair.Detection], now);
        }

        for (var i = 0; i < detections.Count; i++)
        {
            if (result[i] != null) continue;

            var created = new TrackedFace(_nextId++, detections[i], now);
            _active.Add(created);
            result[i] = created;
        }

        return result.Select(t => t!).ToArray();
    }

    // One snapshot per recognized person for the whole video (their tracks are
    // merged), plus one per distinct unknown face.
    public List<SnapshotCandidate> SelectSnapshots(int minUnknownFrames, int maxUnknown)
    {
        var tracks = _finished.Concat(_active).Where(t => t.Best != null).ToList();
        var result = new List<SnapshotCandidate>();

        var known = tracks
            .Select(t => (Track: t, Identity: t.Identity))
            .Where(x => x.Identity != null)
            .GroupBy(x => x.Identity!.Value.EmployeeId);

        foreach (var group in known)
        {
            var best = group.OrderByDescending(x => x.Track.Best!.Quality).First();
            var identity = best.Identity!.Value;
            var vote = best.Track.Votes[identity.EmployeeId];

            result.Add(best.Track.Best! with
            {
                EmployeeId = identity.EmployeeId,
                Name = identity.Name,
                Similarity = vote.BestSimilarity
            });
        }

        var keptUnknown = new List<TrackedFace>();

        foreach (var track in tracks
            .Where(t => t.Identity == null && t.Frames >= minUnknownFrames)
            .OrderByDescending(t => t.Best!.Quality))
        {
            if (keptUnknown.Count >= maxUnknown) break;
            if (keptUnknown.Any(k => Cosine(k.Embedding, track.Embedding) >= MatchSimilarity))
                continue;

            keptUnknown.Add(track);
            result.Add(track.Best! with { EmployeeId = "UNKNOWN", Name = "Unknown" });
        }

        return result;
    }

    public static float Cosine(float[] a, float[] b)
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

    private static double IoU(FaceObservation a, FaceObservation b)
    {
        var x1 = Math.Max(a.X, b.X);
        var y1 = Math.Max(a.Y, b.Y);
        var x2 = Math.Min(a.X + a.Width, b.X + b.Width);
        var y2 = Math.Min(a.Y + a.Height, b.Y + b.Height);

        var intersection = Math.Max(0, x2 - x1) * (double)Math.Max(0, y2 - y1);
        var union = (double)a.Width * a.Height + (double)b.Width * b.Height - intersection;

        return union <= 0 ? 0 : intersection / union;
    }
    public List<TrackedFace> TakeFinished()
    {
        var list = _finished.ToList();
        _finished.Clear();
        return list;
    }
}