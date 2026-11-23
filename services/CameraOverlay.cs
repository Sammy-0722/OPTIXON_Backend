using OpenCvSharp;

namespace AIManagement.Api.services;

public sealed record OverlayBox(int X, int Y, int W, int H, string Label, bool Good);
public sealed record CameraOverlay(List<OverlayBox> Faces, List<OverlayBox> Phones, DateTime At);

public static class OverlayPainter
{
    // Boxes older than 1.5 s are not drawn, so stale boxes never linger
    public static void Draw(Mat img, CameraOverlay? o)
    {
        if (o == null || (DateTime.UtcNow - o.At).TotalSeconds > 1.5) return;
        var thick = Math.Max(2, img.Width / 400);
        var font = Math.Max(0.5, img.Width / 1200.0);

        foreach (var b in o.Faces)
            Box(img, b, b.Good ? new Scalar(80, 200, 80) : new Scalar(60, 60, 230), thick, font);
        foreach (var b in o.Phones)
            Box(img, b, new Scalar(0, 140, 255), thick, font);
    }

    private static void Box(Mat img, OverlayBox b, Scalar c, int thick, double font)
    {
        Cv2.Rectangle(img, new Rect(b.X, b.Y, b.W, b.H), c, thick);
        var size = Cv2.GetTextSize(b.Label, HersheyFonts.HersheySimplex, font, 1, out var bl);
        var top = Math.Max(0, b.Y - size.Height - bl - 4);
        Cv2.Rectangle(img, new Rect(b.X, top, size.Width + 8, size.Height + bl + 4), c, -1);
        Cv2.PutText(img, b.Label, new Point(b.X + 4, top + size.Height + 2),
            HersheyFonts.HersheySimplex, font, Scalar.Black, 1);
    }
}