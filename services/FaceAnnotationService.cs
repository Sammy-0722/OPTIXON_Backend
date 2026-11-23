
using OpenCvSharp;

namespace AIManagement.Api.Services;

public sealed class FaceAnnotationService
{
    public byte[] CreateAnnotatedImage(
        byte[] originalImage,
        IEnumerable<FaceAnnotation> annotations)
    {
        if (annotations == null)
        {
            throw new ArgumentNullException(nameof(annotations));
        }

        using var image = DecodeImage(originalImage);

        foreach (var face in annotations)
        {
            var rectangle = new Rect(
                face.X,
                face.Y,
                face.Width,
                face.Height);

            var label = $"{face.Name} ({face.Similarity:P0})";

            Cv2.Rectangle(
                image,
                rectangle,
                Scalar.LimeGreen,
                2);

            var textSize = Cv2.GetTextSize(
                label,
                HersheyFonts.HersheySimplex,
                0.6,
                2,
                out var baseline);

            var labelY = Math.Max(
                rectangle.Y - 8,
                textSize.Height + baseline + 5);

            var background = new Rect(
                rectangle.X,
                labelY - textSize.Height - baseline - 5,
                Math.Min(
                    textSize.Width + 10,
                    image.Width - rectangle.X),
                textSize.Height + baseline + 10);

            Cv2.Rectangle(
                image,
                background,
                Scalar.LimeGreen,
                -1);

            Cv2.PutText(
                image,
                label,
                new Point(rectangle.X + 5, labelY - 5),
                HersheyFonts.HersheySimplex,
                0.6,
                Scalar.Black,
                2);
        }

        Cv2.ImEncode(".jpg", image, out var outputBytes);

        return outputBytes;
    }

    private static Mat DecodeImage(byte[] imageBytes)
    {
        if (imageBytes == null || imageBytes.Length == 0)
        {
            throw new ArgumentException(
                "The uploaded image is empty.");
        }

        var image = Cv2.ImDecode(
            imageBytes,
            ImreadModes.Color);

        if (image.Empty())
        {
            image.Dispose();

            throw new ArgumentException(
                "The uploaded file is not a valid image.");
        }

        return image;
    }
}

