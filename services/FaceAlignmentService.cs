
using OpenCvSharp;

namespace AIManagement.Api.Services;

public sealed class FaceAlignmentService
{
    private const int FaceSize = 112;

    private static readonly Point2f[] ArcFaceTemplate =
    {
        new(38.2946f, 51.6963f),
        new(73.5318f, 51.5014f),
        new(56.0252f, 71.7366f),
        new(41.5493f, 92.3655f),
        new(70.7299f, 92.2041f)
    };

    public Mat AlignFace(
        Mat image,
        Point2f[] sourcePoints)
    {
        if (image == null || image.Empty())
        {
            throw new ArgumentException(
                "Input image is empty.");
        }

        if (sourcePoints == null || sourcePoints.Length != 5)
        {
            throw new ArgumentException(
                "Exactly five facial landmarks are required.");
        }

        double sourceXMean =
            sourcePoints.Average(p => (double)p.X);

        double sourceYMean =
            sourcePoints.Average(p => (double)p.Y);

        double targetXMean =
            ArcFaceTemplate.Average(p => (double)p.X);

        double targetYMean =
            ArcFaceTemplate.Average(p => (double)p.Y);

        double numeratorA = 0;
        double numeratorB = 0;
        double denominator = 0;

        for (var i = 0; i < 5; i++)
        {
            var sx = sourcePoints[i].X - sourceXMean;
            var sy = sourcePoints[i].Y - sourceYMean;

            var tx = ArcFaceTemplate[i].X - targetXMean;
            var ty = ArcFaceTemplate[i].Y - targetYMean;

            numeratorA += sx * tx + sy * ty;
            numeratorB += sx * ty - sy * tx;

            denominator += sx * sx + sy * sy;
        }

        if (denominator < 1e-8)
        {
            throw new InvalidOperationException(
                "Face landmarks could not be aligned.");
        }

        var a = numeratorA / denominator;
        var b = numeratorB / denominator;

        var txOffset =
            targetXMean - a * sourceXMean + b * sourceYMean;

        var tyOffset =
            targetYMean - b * sourceXMean - a * sourceYMean;

        using var transform =
            new Mat(2, 3, MatType.CV_64FC1);

        transform.Set(0, 0, a);
        transform.Set(0, 1, -b);
        transform.Set(0, 2, txOffset);

        transform.Set(1, 0, b);
        transform.Set(1, 1, a);
        transform.Set(1, 2, tyOffset);

        var aligned = new Mat();

        Cv2.WarpAffine(
            image,
            aligned,
            transform,
            new Size(FaceSize, FaceSize),
            InterpolationFlags.Linear,
            BorderTypes.Constant,
            Scalar.Black);

        return aligned;
    }
}
