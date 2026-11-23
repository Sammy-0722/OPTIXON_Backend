
using OpenCvSharp;

namespace AIManagement.Api.Services;

public sealed class ImageProcessingService
{
    public Mat DecodeImage(byte[] imageBytes)
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

    public Mat ResizeImage(Mat image, int width, int height)
    {
        if (image == null || image.Empty())
        {
            throw new ArgumentException(
                "Input image is empty.");
        }

        if (width <= 0 || height <= 0)
        {
            throw new ArgumentException(
                "Image dimensions must be greater than zero.");
        }

        var resized = new Mat();

        Cv2.Resize(
            image,
            resized,
            new Size(width, height));

        return resized;
    }

    public Mat ConvertBgrToRgb(Mat image)
    {
        if (image == null || image.Empty())
        {
            throw new ArgumentException(
                "Input image is empty.");
        }

        var rgb = new Mat();

        Cv2.CvtColor(
            image,
            rgb,
            ColorConversionCodes.BGR2RGB);

        return rgb;
    }
}
