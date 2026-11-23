
using System.Diagnostics;
using OpenCvSharp;

namespace AIManagement.Api.Services;

public sealed class FaceInferenceService : IDisposable
{
    private readonly ImageProcessingService _imageProcessingService;
    private readonly FaceDetectionService _faceDetectionService;
    private readonly FaceAlignmentService _faceAlignmentService;
    private readonly ArcFaceRecognizer _arcFaceRecognizer;
    private readonly FaceAnnotationService _faceAnnotationService;

    public FaceInferenceService(IWebHostEnvironment environment)
    {
        var modelsDirectory = Path.Combine(
            environment.ContentRootPath, "Models");

        var detectorPath = Path.Combine(
            modelsDirectory, "yolov8n-face.onnx");

        var recognizerPath = Path.Combine(
            modelsDirectory, "w600k_r50.onnx");

        if (!File.Exists(detectorPath))
            throw new FileNotFoundException(
                "YOLO face model was not found.", detectorPath);

        if (!File.Exists(recognizerPath))
            throw new FileNotFoundException(
                "ArcFace model was not found.", recognizerPath);

        _imageProcessingService = new ImageProcessingService();
        _faceDetectionService = new FaceDetectionService(detectorPath);
        _faceAlignmentService = new FaceAlignmentService();
        _arcFaceRecognizer = new ArcFaceRecognizer(recognizerPath);
        _faceAnnotationService = new FaceAnnotationService();
    }

    // YOLO ONLY: Detect faces without generating embeddings.
    public Task<List<DetectedFace>> DetectFacesAsync(
        byte[] imageBytes,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        using var image =
            _imageProcessingService.DecodeImage(imageBytes);

        var faces = _faceDetectionService.DetectFaces(image);

        cancellationToken.ThrowIfCancellationRequested();

        return Task.FromResult(faces);
    }

    // ArcFace ONLY: Generate an embedding for one detected face.
    public Task<float[]> CreateEmbeddingAsync(
        byte[] imageBytes,
        Point2f[] landmarks,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        using var image =
            _imageProcessingService.DecodeImage(imageBytes);

        using var alignedFace =
            _faceAlignmentService.AlignFace(image, landmarks);

        cancellationToken.ThrowIfCancellationRequested();

        var embedding =
            _arcFaceRecognizer.CreateEmbedding(alignedFace);

        return Task.FromResult(embedding);
    }

    // Used when registering one person's face.
    public Task<float[]> GetEmbeddingAsync(
        byte[] imageBytes,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        using var image =
            _imageProcessingService.DecodeImage(imageBytes);

        var faces = _faceDetectionService.DetectFaces(image);

        if (faces.Count == 0)
            throw new InvalidOperationException(
                "No face was detected. Please upload a clear face image.");

        using var alignedFace =
            _faceAlignmentService.AlignFace(image, faces[0].Landmarks);

        var embedding =
            _arcFaceRecognizer.CreateEmbedding(alignedFace);

        return Task.FromResult(embedding);
    }

    // Existing combined operation retained for compatibility.
    public Task<List<FaceInferenceResult>> AnalyzeFacesAsync(
        byte[] imageBytes,
        CancellationToken cancellationToken = default)
    {
        var totalTimer = Stopwatch.StartNew();

        cancellationToken.ThrowIfCancellationRequested();

        var decodeTimer = Stopwatch.StartNew();

        using var image =
            _imageProcessingService.DecodeImage(imageBytes);

        decodeTimer.Stop();

        cancellationToken.ThrowIfCancellationRequested();

        var detectionTimer = Stopwatch.StartNew();

        var faces = _faceDetectionService.DetectFaces(image);

        detectionTimer.Stop();

        var results = new List<FaceInferenceResult>();

        double totalAlignmentMs = 0;
        double totalArcFaceMs = 0;

        foreach (var face in faces)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var alignmentTimer = Stopwatch.StartNew();

            using var alignedFace =
                _faceAlignmentService.AlignFace(image, face.Landmarks);

            alignmentTimer.Stop();

            totalAlignmentMs += alignmentTimer.Elapsed.TotalMilliseconds;

            cancellationToken.ThrowIfCancellationRequested();

            var arcFaceTimer = Stopwatch.StartNew();

            var embedding =
                _arcFaceRecognizer.CreateEmbedding(alignedFace);

            arcFaceTimer.Stop();

            totalArcFaceMs += arcFaceTimer.Elapsed.TotalMilliseconds;

            results.Add(new FaceInferenceResult(
                face.Box.X,
                face.Box.Y,
                face.Box.Width,
                face.Box.Height,
                face.Confidence,
                embedding
            ));
        }

        totalTimer.Stop();

        return Task.FromResult(results);
    }

    // Draw final labels after the repository has identified each face.
    public byte[] CreateAnnotatedImage(
        byte[] originalImage,
        IEnumerable<FaceAnnotation> annotations)
    {
        return _faceAnnotationService.CreateAnnotatedImage(
            originalImage,
            annotations);
    }

    public void Dispose()
    {
        _faceDetectionService.Dispose();
        _arcFaceRecognizer.Dispose();
    }
}
