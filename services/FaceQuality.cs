using OpenCvSharp;

namespace AIManagement.Api.Services;

public static class FaceQuality
{
    // Returns 0..1. Higher = a more useful snapshot.
    // Recognition similarity is deliberately NOT part of this score.
    public static float Score(
        Mat frame, int x, int y, int width, int height, float detectionConfidence)
    {
        var x1 = Math.Max(0, x);
        var y1 = Math.Max(0, y);
        var x2 = Math.Min(frame.Width, x + width);
        var y2 = Math.Min(frame.Height, y + height);
        if (x2 - x1 < 8 || y2 - y1 < 8) return 0;

        using var face = new Mat(frame, new Rect(x1, y1, x2 - x1, y2 - y1));
        using var gray = new Mat();
        Cv2.CvtColor(face, gray, ColorConversionCodes.BGR2GRAY);

        // Fixed size so sharpness is comparable between faces
        using var small = new Mat();
        Cv2.Resize(gray, small, new Size(112, 112));

        // Sharpness / blur: variance of the Laplacian
        using var laplacian = new Mat();
        Cv2.Laplacian(small, laplacian, MatType.CV_64F);
        Cv2.MeanStdDev(laplacian, out _, out var std);
        var sharpness = Math.Clamp(std.Val0 * std.Val0 / 400.0, 0, 1);

        // Exposure: best near mid-gray, worse when too dark or blown out
        var mean = Cv2.Mean(small).Val0;
        var exposure = 1.0 - Math.Min(1.0, Math.Abs(mean - 125) / 125.0);

        // Size: 150 px on the short side counts as full marks
        var size = Math.Clamp(Math.Min(width, height) / 150.0, 0, 1);

        var confidence = Math.Clamp((detectionConfidence - 0.4) / 0.5, 0, 1);

        var score = 0.30 * confidence + 0.30 * sharpness + 0.25 * size + 0.15 * exposure;

        // The detector clamps boxes to the frame, so a face cut off by the edge ends
        // exactly on the border. A cut-off or oddly shaped face is a poor snapshot.
        const int edge = 2;
        var clipped = x <= edge || y <= edge ||
                      x + width >= frame.Width - edge ||
                      y + height >= frame.Height - edge;

        var aspect = width / (double)height;
        var oddShape = aspect < 0.55 || aspect > 1.5;

        if (clipped || oddShape) score *= 0.5;

        return (float)score;
    }
}

public static class FaceCrop
{
    // The face plus margin (hair, shoulders), kept inside the image. Null if it fails.
    public static byte[]? Jpeg(byte[] imageBytes, int x, int y, int width, int height)
    {
        using var image = Cv2.ImDecode(imageBytes, ImreadModes.Color);
        if (image.Empty()) return null;

        var x1 = Math.Max(0, x - (int)(width * 0.6));
        var y1 = Math.Max(0, y - (int)(height * 0.9));
        var x2 = Math.Min(image.Width, x + width + (int)(width * 0.6));
        var y2 = Math.Min(image.Height, y + height + (int)(height * 0.5));
        if (x2 <= x1 || y2 <= y1) return null;

        using var crop = new Mat(image, new Rect(x1, y1, x2 - x1, y2 - y1));
        return crop.ToBytes(".jpg");
    }
}