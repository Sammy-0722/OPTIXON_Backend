using System.Collections.Concurrent;
using System.Linq;
using OpenCvSharp;
using AIManagement.Api.Repositories;

namespace AIManagement.Api.Services;

public class RtspFaceRecognitionService
{
    private readonly FaceInferenceService _inferenceService;
    private readonly FaceRepository _repository;

    private readonly ConcurrentDictionary<string, CameraSession> _cameras = new();

    private const float SimilarityThreshold = 0.40f;
    private const float IoUThreshold = 0.30f;
    private static readonly TimeSpan TrackTimeout = TimeSpan.FromSeconds(3);

    public RtspFaceRecognitionService(
        FaceInferenceService inferenceService,
        FaceRepository repository)
    {
        _inferenceService = inferenceService;
        _repository = repository;
    }

    public bool StartCamera(string cameraId, string rtspUrl)
    {
        var session = new CameraSession(cameraId, rtspUrl);

        if (!_cameras.TryAdd(cameraId, session))
            return false;

        session.ProcessingTask = Task.Run(() => ProcessCameraAsync(session));

        return true;
    }

    public async Task<bool> StopCameraAsync(string cameraId)
    {
        if (!_cameras.TryGetValue(cameraId, out var session))
            return false;

        session.Cancellation.Cancel();

        if (session.ProcessingTask != null)
        {
            try
            {
                await session.ProcessingTask;
            }
            catch (OperationCanceledException)
            {
                // Expected when stopping the camera.
            }
        }

        _cameras.TryRemove(cameraId, out _);
        session.Cancellation.Dispose();

        return true;
    }

    public object? GetCameraStatus(string cameraId)
    {
        if (!_cameras.TryGetValue(cameraId, out var session))
            return null;

        return new
        {
            session.CameraId,
            session.RtspUrl,
            session.Status,
            session.StartedAt
        };
    }

    public object? GetCameraResults(string cameraId)
    {
        if (!_cameras.TryGetValue(cameraId, out var session))
            return null;

        lock (session.Tracks)
        {
            return session.Tracks.Values
                .Select(track => track.Event)
                .OrderByDescending(result => result.LastSeenAt)
                .ToList();
        }
    }

    private async Task ProcessCameraAsync(CameraSession session)
    {
        try
        {
            using var capture = new VideoCapture();

            while (!session.Cancellation.IsCancellationRequested)
            {
                if (!capture.IsOpened())
                {
                    capture.Open(session.RtspUrl);

                    if (!capture.IsOpened())
                    {
                        session.Status = "Reconnecting";
                        await Task.Delay(3000, session.Cancellation.Token);
                        continue;
                    }

                    session.Status = "Connected";
                }

                using var frame = new Mat();

                if (!capture.Read(frame) || frame.Empty())
                {
                    session.Status = "Reconnecting";
                    capture.Release();

                    await Task.Delay(2000, session.Cancellation.Token);
                    continue;
                }

                // Process approximately two frames per second.
                await Task.Delay(500, session.Cancellation.Token);

                Cv2.ImEncode(".jpg", frame, out var imageBytes);

                var faces = await _inferenceService.AnalyzeFacesAsync(
                    imageBytes,
                    session.Cancellation.Token);

                var usedTrackIds = new HashSet<int>();
                var currentTime = DateTime.UtcNow;

                foreach (var face in faces)
                {
                    // Find the best existing track for this face.
                    int? matchedTrackId = FindMatchingTrack(
                        session,
                        face,
                        usedTrackIds,
                        currentTime);

                    // Recognize the face.
                    var match = await _repository.FindMatchAsync(face.Embedding);

                    string name = "Unknown";
                    float similarity = 0;

                    if (match.HasValue &&
                        match.Value.Similarity >= SimilarityThreshold)
                    {
                        name = match.Value.Name;
                        similarity = match.Value.Similarity;
                    }

                    lock (session.Tracks)
                    {
                        if (matchedTrackId.HasValue &&
                            session.Tracks.TryGetValue(
                                matchedTrackId.Value, out var existingTrack))
                        {
                            // Same person: update the existing result.
                            existingTrack.LastSeenAt = currentTime;

                            existingTrack.Event.Name = name;
                            existingTrack.Event.Similarity = similarity;
                            existingTrack.Event.DetectionConfidence =
                                face.DetectionConfidence;

                            existingTrack.Event.X = face.X;
                            existingTrack.Event.Y = face.Y;
                            existingTrack.Event.Width = face.Width;
                            existingTrack.Event.Height = face.Height;

                            existingTrack.Event.LastSeenAt = currentTime;
                            existingTrack.Event.DetectedAt = currentTime;

                            usedTrackIds.Add(matchedTrackId.Value);
                        }
                        else
                        {
                            // New face: create a new track.
                            int newTrackId = ++session.NextTrackId;

                            var faceEvent = new FaceRecognitionEvent
                            {
                                CameraId = session.CameraId,
                                TrackId = newTrackId,
                                Name = name,
                                Similarity = similarity,
                                DetectionConfidence = face.DetectionConfidence,

                                X = face.X,
                                Y = face.Y,
                                Width = face.Width,
                                Height = face.Height,

                                FirstSeenAt = currentTime,
                                LastSeenAt = currentTime,
                                DetectedAt = currentTime
                            };

                            session.Tracks[newTrackId] = new FaceTrack
                            {
                                Event = faceEvent,
                                LastSeenAt = currentTime
                            };

                            usedTrackIds.Add(newTrackId);
                        }
                    }
                }

                // Remove tracks that have not been seen for 3 seconds.
                lock (session.Tracks)
                {
                    var expiredTracks = session.Tracks
                        .Where(item =>
                            currentTime - item.Value.LastSeenAt > TrackTimeout)
                        .Select(item => item.Key)
                        .ToList();

                    foreach (var trackId in expiredTracks)
                    {
                        session.Tracks.Remove(trackId);
                    }

                    // Keep at most 100 active tracks.
                    while (session.Tracks.Count > 100)
                    {
                        var oldestTrack = session.Tracks
                            .OrderBy(item => item.Value.LastSeenAt)
                            .First();

                        session.Tracks.Remove(oldestTrack.Key);
                    }
                }
            }
        }
        catch (OperationCanceledException)
        {
            // Camera was stopped.
        }
        catch (Exception ex)
        {
            session.Status = "Error: " + ex.Message;
        }
        finally
        {
            if (!session.Status.StartsWith("Error:"))
                session.Status = "Stopped";
        }
    }

    private int? FindMatchingTrack(
        CameraSession session,
        FaceInferenceResult face,
        HashSet<int> usedTrackIds,
        DateTime currentTime)
    {
        lock (session.Tracks)
        {
            int? bestTrackId = null;
            float bestIoU = 0;

            foreach (var item in session.Tracks)
            {
                int trackId = item.Key;
                var track = item.Value;

                if (usedTrackIds.Contains(trackId))
                    continue;

                if (currentTime - track.LastSeenAt > TrackTimeout)
                    continue;

                var previousFace = track.Event;

                float iou = CalculateIoU(
                    face.X, face.Y, face.Width, face.Height,
                    previousFace.X, previousFace.Y,
                    previousFace.Width, previousFace.Height);

                if (iou > bestIoU)
                {
                    bestIoU = iou;
                    bestTrackId = trackId;
                }
            }

            if (bestIoU >= IoUThreshold)
                return bestTrackId;

            return null;
        }
    }

    private static float CalculateIoU(
        int x1, int y1, int width1, int height1,
        int x2, int y2, int width2, int height2)
    {
        int left = Math.Max(x1, x2);
        int top = Math.Max(y1, y2);

        int right = Math.Min(x1 + width1, x2 + width2);
        int bottom = Math.Min(y1 + height1, y2 + height2);

        int intersectionWidth = Math.Max(0, right - left);
        int intersectionHeight = Math.Max(0, bottom - top);

        float intersection =
            intersectionWidth * intersectionHeight;

        float area1 = width1 * height1;
        float area2 = width2 * height2;

        float union = area1 + area2 - intersection;

        if (union <= 0)
            return 0;

        return intersection / union;
    }

    private class CameraSession
    {
        public string CameraId { get; }
        public string RtspUrl { get; }

        public string Status { get; set; } = "Starting";

        public DateTime StartedAt { get; } = DateTime.UtcNow;

        public CancellationTokenSource Cancellation { get; } = new();

        public Task? ProcessingTask { get; set; }

        public Dictionary<int, FaceTrack> Tracks { get; } = new();

        public int NextTrackId { get; set; } = 0;

        public CameraSession(string cameraId, string rtspUrl)
        {
            CameraId = cameraId;
            RtspUrl = rtspUrl;
        }
    }

    private class FaceTrack
    {
        public FaceRecognitionEvent Event { get; set; } = new();

        public DateTime LastSeenAt { get; set; }
    }
}

public class FaceRecognitionEvent
{
    public string CameraId { get; set; } = "";

    public int TrackId { get; set; }

    public string Name { get; set; } = "";

    public float Similarity { get; set; }

    public float DetectionConfidence { get; set; }

    public int X { get; set; }

    public int Y { get; set; }

    public int Width { get; set; }

    public int Height { get; set; }

    public DateTime FirstSeenAt { get; set; }

    public DateTime LastSeenAt { get; set; }

    public DateTime DetectedAt { get; set; }
}