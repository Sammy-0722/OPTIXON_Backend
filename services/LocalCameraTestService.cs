
using OpenCvSharp;
using AIManagement.Api.Repositories;
using System.Diagnostics;

namespace AIManagement.Api.Services;

public sealed class LocalCameraTestService : IDisposable
{
    private readonly object _lock = new();

    private readonly FaceInferenceService _faceInferenceService;
    private readonly FaceRepository _faceRepository;

    private VideoCapture? _capture;
    private CancellationTokenSource? _cts;
    private Task? _captureTask;

    private byte[]? _latestFrame;
    private bool _isRunning;

    public LocalCameraTestService(
        FaceInferenceService faceInferenceService,
        FaceRepository faceRepository)
    {
        _faceInferenceService = faceInferenceService;
        _faceRepository = faceRepository;
    }

    public bool IsRunning
    {
        get
        {
            lock (_lock)
                return _isRunning;
        }
    }

    public byte[]? GetLatestFrame()
    {
        lock (_lock)
            return _latestFrame?.ToArray();
    }

    public Task StartAsync()
    {
        lock (_lock)
        {
            if (_isRunning)
                return Task.CompletedTask;

            // Open the physical webcam on the API computer.
            var capture = new VideoCapture(0, VideoCaptureAPIs.DSHOW);

            if (!capture.IsOpened())
            {
                capture.Dispose();
                throw new InvalidOperationException(
                    "Unable to open webcam. Check whether another application is using it.");
            }

            capture.Set(VideoCaptureProperties.FrameWidth, 640);
            capture.Set(VideoCaptureProperties.FrameHeight, 480);

            _capture = capture;
            _cts = new CancellationTokenSource();
            _isRunning = true;

            _captureTask = Task.Run(
                () => CaptureLoop(_cts.Token));

            return Task.CompletedTask;
        }
    }



    private void CaptureLoop(CancellationToken cancellationToken)
    {
        using var frame = new Mat();

        const string rawWindow = "AI Management - Live Camera";
        const string annotatedWindow = "AI Management - Face Recognition";

        double fps = 0;
        const double smoothing = 0.2;

        Cv2.NamedWindow(rawWindow, WindowFlags.Normal);
        Cv2.NamedWindow(annotatedWindow, WindowFlags.Normal);

        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                var frameTimer = Stopwatch.StartNew();

                VideoCapture? capture;

                lock (_lock)
                    capture = _capture;

                if (capture == null)
                    break;

                if (!capture.Read(frame) || frame.Empty())
                    continue;

                Cv2.ImEncode(".jpg", frame, out byte[] imageBytes);

                try
                {
                    var results = _faceInferenceService
                        .AnalyzeFacesAsync(imageBytes, cancellationToken)
                        .GetAwaiter()
                        .GetResult();

                    var annotations = new List<FaceAnnotation>();

                    foreach (var face in results)
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        var match = _faceRepository
                            .FindMatchAsync(face.Embedding)
                            .GetAwaiter()
                            .GetResult();

                        const float threshold = 0.40f;

                        bool recognized =
                            match.HasValue &&
                            match.Value.Similarity >= threshold;

                        string name = recognized
                            ? match.Value.Name
                            : "Unknown";

                        float similarity = recognized
                            ? match.Value.Similarity
                            : 0f;

                        annotations.Add(new FaceAnnotation(
                            face.X,
                            face.Y,
                            face.Width,
                            face.Height,
                            name,
                            similarity
                        ));
                    }

                    byte[] annotatedImage =
                        _faceInferenceService.CreateAnnotatedImage(
                            imageBytes,
                            annotations);

                    lock (_lock)
                        _latestFrame = annotatedImage;

                    using var annotatedMat = Cv2.ImDecode(
                        annotatedImage,
                        ImreadModes.Color);

                    frameTimer.Stop();

                    double currentFps = 1000.0 / Math.Max(
                        frameTimer.Elapsed.TotalMilliseconds, 1);

                    fps = fps == 0
                        ? currentFps
                        : fps * (1 - smoothing) + currentFps * smoothing;

                    // FPS on original camera.
                    Cv2.PutText(
                        frame,
                        $"Camera FPS: {fps:F1}",
                        new Point(20, 40),
                        HersheyFonts.HersheySimplex,
                        1,
                        Scalar.LimeGreen,
                        2);

                    // FPS on annotated camera.
                    if (!annotatedMat.Empty())
                    {
                        Cv2.PutText(
                            annotatedMat,
                            $"Recognition FPS: {fps:F1}",
                            new Point(20, 40),
                            HersheyFonts.HersheySimplex,
                            1,
                            Scalar.LimeGreen,
                            2);

                        Cv2.ImShow(annotatedWindow, annotatedMat);
                    }

                    Cv2.ImShow(rawWindow, frame);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Console.WriteLine(
                        $"Camera processing error: {ex.Message}");
                }

                int key = Cv2.WaitKey(1);

                if (key == 27)
                    break;
            }
        }
        finally
        {
            lock (_lock)
            {
                _capture?.Release();
                _capture?.Dispose();
                _capture = null;

                _latestFrame = null;
                _isRunning = false;
            }

            Cv2.DestroyWindow(rawWindow);
            Cv2.DestroyWindow(annotatedWindow);

            Console.WriteLine("Local camera test stopped.");
        }
    }



    public async Task StopAsync()
    {
        Task? task;

        lock (_lock)
        {
            _cts?.Cancel();
            task = _captureTask;
        }

        if (task != null)
            await task;

        lock (_lock)
        {
            _cts?.Dispose();
            _cts = null;
            _captureTask = null;
            _isRunning = false;
        }
    }

    public void Dispose()
    {
        StopAsync().GetAwaiter().GetResult();
    }
}
