using AIManagement.Api.services;
using Microsoft.Extensions.Options;

namespace AIManagement.Api.Services;

public sealed class StickyIdentityOptions
{
    public bool Enabled { get; set; } = true;

    // How long a face may be missing before it counts as having left the frame
    public double GraceSeconds { get; set; } = 4.0;

    // A first recognition at or above this locks the name
    public float LockSimilarity { get; set; } = 0.40f;

    // A different person must match at least this well to replace a locked name
    public float OverrideSimilarity { get; set; } = 0.55f;
}

public sealed record StickyFace(
    int X, int Y, int Width, int Height, float[] Embedding,
    bool Recognized, string EmployeeId, string Name, float Similarity);

public sealed record StickyResult(
    bool Recognized, string EmployeeId, string Name, float Similarity);

public sealed class StickyIdentityService
{
    private sealed class LockedIdentity
    {
        public string EmployeeId = "";
        public string Name = "";
        public float Similarity;   // last real similarity of the locked person
    }

    private readonly StickyIdentityOptions _options;
    private readonly object _gate = new();
    private readonly Dictionary<string, FaceTracker> _trackers = new();
    private readonly Dictionary<string, LockedIdentity> _locks = new();

    public StickyIdentityService(IOptions<StickyIdentityOptions> options)
    {
        _options = options.Value;
    }

    // Returns one result per face, in the same order. Call it once per frame,
    // also when the frame has no faces, so that faces that left are released.
    public IReadOnlyList<StickyResult> Resolve(string streamKey, IReadOnlyList<StickyFace> faces)
    {
        if (!_options.Enabled)
        {
            return faces
                .Select(f => new StickyResult(f.Recognized, f.EmployeeId, f.Name, f.Similarity))
                .ToList();
        }

        lock (_gate)
        {
            if (!_trackers.TryGetValue(streamKey, out var tracker))
                _trackers[streamKey] = tracker = new FaceTracker(_options.GraceSeconds);

            var now = Environment.TickCount64 / 1000.0;

            var observations = faces
                .Select(f => new FaceObservation(f.X, f.Y, f.Width, f.Height, f.Embedding))
                .ToList();

            // Links every face to the same face's track from the earlier frames
            var tracks = tracker.Associate(observations, now);
            var results = new List<StickyResult>(faces.Count);

            for (var i = 0; i < faces.Count; i++)
            {
                var face = faces[i];
                var key = $"{streamKey}#{tracks[i].TrackId}";
                _locks.TryGetValue(key, out var locked);

                if (face.Recognized)
                {
                    if (locked == null)
                    {
                        if (face.Similarity >= _options.LockSimilarity)
                        {
                            _locks[key] = locked = new LockedIdentity
                            {
                                EmployeeId = face.EmployeeId,
                                Name = face.Name,
                                Similarity = face.Similarity
                            };
                        }
                    }
                    else if (locked.EmployeeId == face.EmployeeId)
                    {
                        locked.Similarity = face.Similarity;
                    }
                    else if (face.Similarity >= _options.OverrideSimilarity)
                    {
                        // A different person matches strongly: the old lock was wrong
                        locked.EmployeeId = face.EmployeeId;
                        locked.Name = face.Name;
                        locked.Similarity = face.Similarity;
                    }
                    // else: a weak match for someone else, so the locked name stays
                }

                if (locked != null)
                    results.Add(new StickyResult(true, locked.EmployeeId, locked.Name, locked.Similarity));
                else if (face.Recognized)
                    results.Add(new StickyResult(true, face.EmployeeId, face.Name, face.Similarity));
                else
                    results.Add(new StickyResult(false, "UNKNOWN", "Unknown", face.Similarity));
            }

            // Faces that left the frame lose their locked name
            foreach (var ended in tracker.TakeFinished())
                _locks.Remove($"{streamKey}#{ended.TrackId}");

            return results;
        }
    }
}