

using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace AIManagement.Api.Services;

public sealed class FaceDetectionService : IDisposable
{
    private const int DetectorSize = 320;
    private const float ConfidenceThreshold = 0.30f;
    private const float NmsThreshold = 0.45f;

    private readonly InferenceSession _detector;

    public FaceDetectionService(string modelPath)
    {
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException(
                "YOLO face model was not found.", modelPath);
        }

        using var detectorOptions =
            new Microsoft.ML.OnnxRuntime.SessionOptions();

        detectorOptions.AppendExecutionProvider_DML(0);
        detectorOptions.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
        detectorOptions.EnableMemoryPattern = false;

        _detector = new InferenceSession(
            modelPath,
            detectorOptions);

        Console.WriteLine("YOLO initialized with DirectML.");
    }

    public List<DetectedFace> DetectFaces(Mat image)
    {
        if (image == null || image.Empty())
        {
            throw new ArgumentException(
                "Input image is empty.");
        }

        using var resized = new Mat();

        Cv2.Resize(
            image,
            resized,
            new Size(DetectorSize, DetectorSize));

        using var rgb = new Mat();

        Cv2.CvtColor(
            resized,
            rgb,
            ColorConversionCodes.BGR2RGB);

        var input = new DenseTensor<float>(
            new[] { 1, 3, DetectorSize, DetectorSize });

        var pixels = new byte[DetectorSize * DetectorSize * 3];
        Marshal.Copy(rgb.Data, pixels, 0, pixels.Length);

        var span = input.Buffer.Span;
        var plane = DetectorSize * DetectorSize;

        for (var i = 0; i < plane; i++)
        {
            span[i] = pixels[i * 3] / 255f;
            span[plane + i] = pixels[i * 3 + 1] / 255f;
            span[2 * plane + i] = pixels[i * 3 + 2] / 255f;
        }

        using var results = _detector.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor("images", input)
        });

        var output = results.First().AsTensor<float>();
        var dimensions = output.Dimensions.ToArray();

        if (dimensions.Length != 3 ||
            dimensions[1] < 20 ||
            dimensions[2] == 0)
        {
            throw new InvalidOperationException(
                "Unexpected YOLO output shape. Expected [1,20,2100].");
        }

        var detections = dimensions[2];

        float GetValue(int channel, int detection) =>
            output[0, channel, detection];

        var scaleX = (float)image.Width / DetectorSize;
        var scaleY = (float)image.Height / DetectorSize;

        var candidates = new List<DetectedFace>();

        for (var i = 0; i < detections; i++)
        {
            var confidence = GetValue(4, i);

            if (confidence < ConfidenceThreshold)
                continue;

            var centerX = GetValue(0, i);
            var centerY = GetValue(1, i);
            var width = GetValue(2, i);
            var height = GetValue(3, i);

            var left = (centerX - width / 2) * scaleX;
            var top = (centerY - height / 2) * scaleY;
            var boxWidth = width * scaleX;
            var boxHeight = height * scaleY;

            var x1 = Math.Clamp(
                (int)left, 0, image.Width - 1);

            var y1 = Math.Clamp(
                (int)top, 0, image.Height - 1);

            var x2 = Math.Clamp(
                (int)(left + boxWidth), 0, image.Width);

            var y2 = Math.Clamp(
                (int)(top + boxHeight), 0, image.Height);

            if (x2 <= x1 || y2 <= y1)
                continue;

            var landmarks = new Point2f[5];

            for (var landmark = 0; landmark < 5; landmark++)
            {
                var lx =
                    GetValue(5 + landmark * 3, i) * scaleX;

                var ly =
                    GetValue(6 + landmark * 3, i) * scaleY;

                landmarks[landmark] = new Point2f(lx, ly);
            }

            candidates.Add(new DetectedFace(
                new Rect(x1, y1, x2 - x1, y2 - y1),
                confidence,
                landmarks));
        }

        return ApplyNms(candidates);
    }

    private static List<DetectedFace> ApplyNms(
        List<DetectedFace> candidates)
    {
        var ordered = candidates
            .OrderByDescending(c => c.Confidence)
            .ToList();

        var kept = new List<DetectedFace>();

        foreach (var candidate in ordered)
        {
            if (kept.All(existing =>
                CalculateIoU(
                    candidate.Box,
                    existing.Box) <= NmsThreshold))
            {
                kept.Add(candidate);
            }
        }

        return kept;
    }

    private static float CalculateIoU(Rect a, Rect b)
    {
        var x1 = Math.Max(a.X, b.X);
        var y1 = Math.Max(a.Y, b.Y);
        var x2 = Math.Min(a.Right, b.Right);
        var y2 = Math.Min(a.Bottom, b.Bottom);

        var intersectionWidth = Math.Max(0, x2 - x1);
        var intersectionHeight = Math.Max(0, y2 - y1);

        var intersection =
            intersectionWidth * intersectionHeight;

        var union =
            a.Width * a.Height +
            b.Width * b.Height -
            intersection;

        return union <= 0
            ? 0
            : (float)intersection / union;
    }

    public void Dispose()
    {
        _detector.Dispose();
    }
}

public sealed record DetectedFace(
    Rect Box,
    float Confidence,
    Point2f[] Landmarks);
