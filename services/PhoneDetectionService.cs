using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.Runtime.InteropServices;

namespace AIManagement.Api.services;

public sealed class PhoneDetectionOptions
{
    public float MinConfidence { get; set; } = 0.35f;
    public float PersonConfidence { get; set; } = 0.30f;
    

    
}

public sealed record PhoneDetection(int X, int Y, int Width, int Height, float Confidence);

// A person box plus their hand points (wrist and a point just past the wrist)
public sealed record PersonPose(PhoneDetection Box, List<(float X, float Y)> Hands);

public sealed class PhoneDetectionService : IDisposable
{
    private const int InputSize = 640;
    private const int PhoneClass = 67;               // "cell phone" in COCO
    private const float NmsThreshold = 0.45f;

    // Pose keypoint indexes (COCO 17)
    private const int LeftElbow = 7, RightElbow = 8, LeftWrist = 9, RightWrist = 10;
    private const float KeypointMinConfidence = 0.2f; // low on purpose: twisted wrists score lower
    private const float HandExtension = 0.35f;        // hand point = wrist + 35% of the forearm

    private readonly InferenceSession _detector;
    private readonly InferenceSession _pose;
    private readonly string _detectorInput;
    private readonly string _poseInput;
    private readonly float _minConfidence;
    private readonly float _personConfidence;

    public PhoneDetectionService(
        IWebHostEnvironment environment,
        IOptions<PhoneDetectionOptions> options)
    {
        var models = Path.Combine(environment.ContentRootPath, "Models");
        var detectorPath = Path.Combine(models, "yolov8n.onnx");
        var posePath = Path.Combine(models, "yolov8n-pose.onnx");

        if (!File.Exists(detectorPath))
            throw new FileNotFoundException("Phone detection model was not found.", detectorPath);
        if (!File.Exists(posePath))
            throw new FileNotFoundException(
                "Pose model was not found. Export yolov8n-pose with imgsz=640.", posePath);

        _minConfidence = options.Value.MinConfidence;
        _personConfidence = options.Value.PersonConfidence;

        _detector = CreateSession(detectorPath);
        _pose = CreateSession(posePath);
        _detectorInput = _detector.InputMetadata.Keys.First();
        _poseInput = _pose.InputMetadata.Keys.First();
    }

    private static InferenceSession CreateSession(string modelPath)
    {
        try
        {
            using var gpu = new Microsoft.ML.OnnxRuntime.SessionOptions();
            gpu.AppendExecutionProvider_DML(0);
            gpu.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            gpu.EnableMemoryPattern = false;
            return new InferenceSession(modelPath, gpu);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Phone detector: DirectML unavailable ({ex.Message}), using CPU.");
            return new InferenceSession(modelPath);
        }
    }
    private static List<PhoneDetection> DropContained(List<PhoneDetection> boxes)
    {
        var kept = new List<PhoneDetection>();

        foreach (var b in boxes.OrderByDescending(x => x.Confidence))
        {
            var duplicate = kept.Any(k =>
            {
                var x1 = Math.Max(k.X, b.X);
                var y1 = Math.Max(k.Y, b.Y);
                var x2 = Math.Min(k.X + k.Width, b.X + b.Width);
                var y2 = Math.Min(k.Y + k.Height, b.Y + b.Height);
                var inter = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
                var smaller = Math.Min(k.Width * k.Height, b.Width * b.Height);
                return smaller > 0 && (float)inter / smaller > 0.5f;
            });

            if (!duplicate) kept.Add(b);
        }

        return kept;
    }

    // Kept for older callers
    public List<PhoneDetection> Detect(Mat image) => DetectPhones(image);

    // ---- Letterbox + tensor ----
    private static (DenseTensor<float> Input, float Scale, int PadX, int PadY) Prepare(Mat image)
    {
        if (image == null || image.Empty())
            throw new ArgumentException("Input image is empty.");

        var scale = Math.Min((float)InputSize / image.Width, (float)InputSize / image.Height);
        var newWidth = (int)Math.Round(image.Width * scale);
        var newHeight = (int)Math.Round(image.Height * scale);
        var padX = (InputSize - newWidth) / 2;
        var padY = (InputSize - newHeight) / 2;

        using var resized = new Mat();
        Cv2.Resize(image, resized, new Size(newWidth, newHeight));

        using var canvas = new Mat(
            new Size(InputSize, InputSize), MatType.CV_8UC3, new Scalar(114, 114, 114));
        using (var roi = new Mat(canvas, new Rect(padX, padY, newWidth, newHeight)))
        {
            resized.CopyTo(roi);
        }

        using var rgb = new Mat();
        Cv2.CvtColor(canvas, rgb, ColorConversionCodes.BGR2RGB);

        var pixels = new byte[InputSize * InputSize * 3];
        Marshal.Copy(rgb.Data, pixels, 0, pixels.Length);

        var input = new DenseTensor<float>(new[] { 1, 3, InputSize, InputSize });
        var span = input.Buffer.Span;
        var plane = InputSize * InputSize;

        for (var i = 0; i < plane; i++)
        {
            span[i] = pixels[i * 3] / 255f;
            span[plane + i] = pixels[i * 3 + 1] / 255f;
            span[2 * plane + i] = pixels[i * 3 + 2] / 255f;
        }

        return (input, scale, padX, padY);
    }

    private static (float[] Data, int Cols) Run(
        InferenceSession session, string inputName, DenseTensor<float> input, int expectedRows)
    {
        using var results = session.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor(inputName, input)
        });

        var output = results.First().AsTensor<float>();
        var dims = output.Dimensions.ToArray();

        if (dims.Length != 3 || dims[1] != expectedRows)
            throw new InvalidOperationException(
                $"Unexpected model output (expected [1,{expectedRows},8400]). Export with imgsz=640.");

        return (output.ToArray(), dims[2]);
    }

    private static PhoneDetection? ToBox(
        float[] d, int cols, int i, float score, float scale, int padX, int padY, Mat image)
    {
        var cx = d[0 * cols + i];
        var cy = d[1 * cols + i];
        var w = d[2 * cols + i];
        var h = d[3 * cols + i];

        var x1 = Math.Clamp((int)((cx - w / 2 - padX) / scale), 0, image.Width - 1);
        var y1 = Math.Clamp((int)((cy - h / 2 - padY) / scale), 0, image.Height - 1);
        var x2 = Math.Clamp((int)((cx + w / 2 - padX) / scale), 0, image.Width);
        var y2 = Math.Clamp((int)((cy + h / 2 - padY) / scale), 0, image.Height);

        if (x2 - x1 < 4 || y2 - y1 < 4) return null;
        return new PhoneDetection(x1, y1, x2 - x1, y2 - y1, score);
    }

    // ---- Phones (YOLOv8 detection, class 67) ----
    public List<PhoneDetection> DetectPhones(Mat image)
    {
        var (input, scale, padX, padY) = Prepare(image);
        var (data, cols) = Run(_detector, _detectorInput, input, 84);

        var found = new List<PhoneDetection>();
        for (var i = 0; i < cols; i++)
        {
            var score = data[(4 + PhoneClass) * cols + i];
            if (score < _minConfidence) continue;

            var box = ToBox(data, cols, i, score, scale, padX, padY, image);
            if (box != null) found.Add(box);
        }

        return DropContained(Nms(found, b => b));
    }

    // ---- People with hand points (YOLOv8 pose) ----
    public List<PersonPose> DetectPeople(Mat image)
    {
        var (input, scale, padX, padY) = Prepare(image);
        var (data, cols) = Run(_pose, _poseInput, input, 56);

        var found = new List<PersonPose>();
        for (var i = 0; i < cols; i++)
        {
            var score = data[4 * cols + i];
            if (score < _personConfidence) continue;

            var box = ToBox(data, cols, i, score, scale, padX, padY, image);
            if (box == null) continue;

            var hands = new List<(float X, float Y)>();

            foreach (var (wrist, elbow) in new[] { (LeftWrist, LeftElbow), (RightWrist, RightElbow) })
            {
                var wc = data[(5 + wrist * 3 + 2) * cols + i];
                if (wc < KeypointMinConfidence) continue;

                var wx = (data[(5 + wrist * 3) * cols + i] - padX) / scale;
                var wy = (data[(5 + wrist * 3 + 1) * cols + i] - padY) / scale;
                hands.Add((wx, wy));

                // A second point just past the wrist along the forearm: still lands on the hand
                // when the wrist is twisted and the back of the hand faces the camera
                var ec = data[(5 + elbow * 3 + 2) * cols + i];
                if (ec >= KeypointMinConfidence)
                {
                    var ex = (data[(5 + elbow * 3) * cols + i] - padX) / scale;
                    var ey = (data[(5 + elbow * 3 + 1) * cols + i] - padY) / scale;
                    hands.Add((wx + HandExtension * (wx - ex), wy + HandExtension * (wy - ey)));
                }
            }

            found.Add(new PersonPose(box, hands));
        }

        return Nms(found, p => p.Box);
    }

    // Draws a box and label for each phone and returns a JPEG
    public byte[] Annotate(Mat image, IReadOnlyList<PhoneDetection> phones)
    {
        using var canvas = image.Clone();

        var thickness = Math.Max(2, canvas.Width / 400);
        var fontScale = Math.Max(0.5, canvas.Width / 1200.0);
        var color = new Scalar(0, 140, 255); // orange (BGR)

        foreach (var phone in phones)
        {
            Cv2.Rectangle(canvas, new Rect(phone.X, phone.Y, phone.Width, phone.Height), color, thickness);

            var label = $"phone in hand {phone.Confidence * 100:0}%";
            var size = Cv2.GetTextSize(
                label, HersheyFonts.HersheySimplex, fontScale, 1, out var baseline);

            var top = Math.Max(0, phone.Y - size.Height - baseline - 4);
            Cv2.Rectangle(
                canvas, new Rect(phone.X, top, size.Width + 8, size.Height + baseline + 4), color, -1);
            Cv2.PutText(
                canvas, label, new Point(phone.X + 4, top + size.Height + 2),
                HersheyFonts.HersheySimplex, fontScale, Scalar.Black, 1);
        }

        return canvas.ToBytes(".jpg");
    }

    private static List<T> Nms<T>(List<T> items, Func<T, PhoneDetection> box)
    {
        var kept = new List<T>();

        foreach (var item in items.OrderByDescending(c => box(c).Confidence))
        {
            if (kept.All(k => IoU(box(item), box(k)) <= NmsThreshold))
                kept.Add(item);
        }

        return kept;
    }

    public static float IoU(PhoneDetection a, PhoneDetection b)
    {
        var x1 = Math.Max(a.X, b.X);
        var y1 = Math.Max(a.Y, b.Y);
        var x2 = Math.Min(a.X + a.Width, b.X + b.Width);
        var y2 = Math.Min(a.Y + a.Height, b.Y + b.Height);

        var intersection = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
        var union = a.Width * a.Height + b.Width * b.Height - intersection;

        return union <= 0 ? 0 : (float)intersection / union;
    }

    public void Dispose()
    {
        _detector.Dispose();
        _pose.Dispose();
    }
}

public enum PhoneStatus
{
    NotInHand,  // no hand is near the phone
    InHand,     // a hand point is on or near the phone, held low
    InUse       // a hand point is on or near the phone, at chest or face height
}

public sealed record PhoneUse(PhoneDetection Phone, PhoneStatus Status, int? PersonIndex);

public sealed record PhoneAnalysis(
    List<PhoneUse> Phones,
    List<PhoneDetection> People);

public sealed class PhoneUseService
{
    // A hand point counts when it is within this many phone-lengths of the phone box
    private const float ReachInPhoneLengths = 0.6f;

    // A phone whose centre is in the top part of the person's box is likely being used
    private const float InUseMaxRelativeY = 0.6f;

    private readonly PhoneDetectionService _detector;

    public PhoneUseService(PhoneDetectionService detector)
    {
        _detector = detector;
    }
    private readonly object _gate = new();

    public PhoneAnalysis Analyze(Mat frame)

    {
        lock (_gate)
        {
            var phones = _detector.DetectPhones(frame);
            var people = _detector.DetectPeople(frame);
            var uses = new List<PhoneUse>();

            foreach (var phone in phones)
            {
                var reach = Math.Max(phone.Width, phone.Height) * ReachInPhoneLengths;
                int? owner = null;
                var bestDistance = float.MaxValue;

                for (var i = 0; i < people.Count; i++)
                {
                    foreach (var hand in people[i].Hands)
                    {
                        var distance = DistanceToBox(hand.X, hand.Y, phone);
                        if (distance <= reach && distance < bestDistance)
                        {
                            bestDistance = distance;
                            owner = i;
                        }
                    }
                }

                if (owner is null)
                {
                    uses.Add(new PhoneUse(phone, PhoneStatus.NotInHand, null));
                    continue;
                }

                var person = people[owner.Value].Box;
                var relativeY = (phone.Y + phone.Height / 2f - person.Y) / person.Height;
                var status = relativeY <= InUseMaxRelativeY ? PhoneStatus.InUse : PhoneStatus.InHand;

                uses.Add(new PhoneUse(phone, status, owner));
            }

            return new PhoneAnalysis(uses, people.Select(p => p.Box).ToList());
        }
    }

    // 0 when the point is inside the box, otherwise the distance to its nearest edge
    private static float DistanceToBox(float px, float py, PhoneDetection box)
    {
        var dx = Math.Max(Math.Max(box.X - px, 0), px - (box.X + box.Width));
        var dy = Math.Max(Math.Max(box.Y - py, 0), py - (box.Y + box.Height));
        return MathF.Sqrt(dx * dx + dy * dy);
    }
    // Drops a box that mostly sits inside a stronger box (the same phone detected twice)
  
}