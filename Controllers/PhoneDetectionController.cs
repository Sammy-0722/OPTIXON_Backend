using AIManagement.Api.services;
using AIManagement.Api.Services;
using Microsoft.AspNetCore.Mvc;
using OpenCvSharp;

namespace AIManagement.Api.Controllers;

[ApiController]
[Route("api/[controller]")]
public class PhoneDetectionController : ControllerBase
{
    private readonly PhoneDetectionService _phones;
    private readonly PhoneUseService _phoneUse;
    private readonly StreamHub _streamHub;
    private readonly PhoneSnapshotService _snapshots;

    public PhoneDetectionController(
        PhoneDetectionService phones,
        PhoneUseService phoneUse,
        StreamHub streamHub,
        PhoneSnapshotService snapshots)
    {
        _phones = phones;
        _phoneUse = phoneUse;
        _streamHub = streamHub;
        _snapshots = snapshots;
    }

    // ---------------------------------------------------------------
    // DETECT PHONES IN AN IMAGE
    // ---------------------------------------------------------------
    [HttpPost("detect")]
    [Consumes("multipart/form-data")]
    public async Task<IActionResult> Detect(IFormFile image, CancellationToken cancellationToken)
    {
        if (image == null || image.Length == 0)
            return BadRequest("Please upload an image.");

        if (image.Length > 10 * 1024 * 1024)
            return BadRequest("Image must be smaller than 10 MB.");

        await using var stream = new MemoryStream();
        await image.CopyToAsync(stream, cancellationToken);

        using var mat = Cv2.ImDecode(stream.ToArray(), ImreadModes.Color);
        if (mat.Empty())
            return BadRequest("The uploaded file is not a valid image.");

        try
        {
            var analysis = _phoneUse.Analyze(mat);
            Console.WriteLine($"[phone] phones={analysis.Phones.Count} " +
    $"inHand={analysis.Phones.Count(u => u.Status != PhoneStatus.NotInHand)} people={analysis.People.Count}");
            var held = analysis.Phones.Where(u => u.Status != PhoneStatus.NotInHand).ToList();
            var annotated = _phones.Annotate(mat, held.Select(u => u.Phone).ToList());

            return Ok(new
            {
                message = "Phone detection completed.",
                totalPhones = held.Count,
                phonesInHand = held.Count,
                detections = held.Select(u => new
                {
                    x = u.Phone.X,
                    y = u.Phone.Y,
                    width = u.Phone.Width,
                    height = u.Phone.Height,
                    confidence = u.Phone.Confidence,
                    status = u.Status.ToString(),
                    personIndex = u.PersonIndex
                }),
                imageBase64 = Convert.ToBase64String(annotated)
            });
        }
        catch (InvalidOperationException ex)
        {
            return BadRequest(new { message = "Phone detection failed.", error = ex.Message });
        }
    }

    // ---------------------------------------------------------------
    // DETECT PHONES IN A VIDEO
    // ---------------------------------------------------------------

    // ---------------------------------------------------------------
    // DETECT PHONES IN A VIDEO
    // ---------------------------------------------------------------
    [HttpPost("detect-video")]
    [Consumes("multipart/form-data")]
    [RequestSizeLimit(100 * 1024 * 1024)]
    [RequestFormLimits(MultipartBodyLengthLimit = 100 * 1024 * 1024)]
    public async Task<IActionResult> DetectVideo(IFormFile video, CancellationToken cancellationToken)
    {
        if (video == null || video.Length == 0)
            return BadRequest("Please upload a video.");

        if (video.Length > 100 * 1024 * 1024)
            return BadRequest("Video must be smaller than 100 MB.");

        var allowedExtensions = new[] { ".mp4", ".avi", ".mov", ".mkv", ".webm" };
        var extension = Path.GetExtension(video.FileName).ToLowerInvariant();

        if (!allowedExtensions.Contains(extension))
            return BadRequest("Unsupported video format.");

        var tempFile = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}{extension}");

        try
        {
            await using (var fileStream = new FileStream(tempFile, FileMode.Create, FileAccess.Write))
            {
                await video.CopyToAsync(fileStream, cancellationToken);
            }

            using var capture = new VideoCapture(tempFile);

            if (!capture.IsOpened())
                return BadRequest("Unable to open the uploaded video.");

            var fps = capture.Fps;
            if (fps <= 0 || double.IsNaN(fps) || double.IsInfinity(fps))
                fps = 25;

            // One frame is analysed every half second
            const double SampleEverySeconds = 0.5;
            var frameInterval = (long)Math.Max(1, fps * SampleEverySeconds);

            using var frame = new Mat();

            long frameNumber = 0;
            var analysed = 0;
            var framesWithPhone = 0;
            var framesWithPhoneInHand = 0;
            var maxPhones = 0;
            double firstSeen = -1;
            double lastSeen = -1;

            byte[]? bestJpeg = null;
            var bestCount = 0;
            var bestConfidence = 0f;
            double bestTime = 0;

            var frames = new List<object>();
            var recentHeld = new List<(PhoneDetection Box, double Time)>();

            while (true)
            {
                cancellationToken.ThrowIfCancellationRequested();

                if (frameNumber % frameInterval != 0)
                {
                    if (!capture.Grab()) break;
                    frameNumber++;
                    continue;
                }

                if (!capture.Read(frame) || frame.Empty())
                    break;

                var timestamp = frameNumber / fps;
                var analysis = _phoneUse.Analyze(frame);
                recentHeld.RemoveAll(h => timestamp - h.Time > 2.0);

                var phones = analysis.Phones
                    .Select(u => u.Status != PhoneStatus.NotInHand
                        ? u
                        : recentHeld.Any(h => PhoneDetectionService.IoU(h.Box, u.Phone) > 0.3f)
                            ? u with { Status = PhoneStatus.InHand }
                            : null)
                    .Where(u => u != null)
                    .Select(u => u!)
                    .ToList();

                foreach (var u in phones) recentHeld.Add((u.Phone, timestamp));
                analysed++;

                if (phones.Count > 0)
                {
                    framesWithPhone++;
                    if (phones.Any(u => u.Status != PhoneStatus.NotInHand)) framesWithPhoneInHand++;

                    maxPhones = Math.Max(maxPhones, phones.Count);
                    if (firstSeen < 0) firstSeen = timestamp;
                    lastSeen = timestamp;

                    var confidence = phones.Max(u => u.Phone.Confidence);

                    if (phones.Count > bestCount ||
                        (phones.Count == bestCount && confidence > bestConfidence))
                    {
                        bestCount = phones.Count;
                        bestConfidence = confidence;
                        bestTime = timestamp;
                        bestJpeg = _phones.Annotate(frame, phones.Select(u => u.Phone).ToList());
                    }
                }

                frames.Add(new
                {
                    timestamp,
                    detections = phones.Select(u => new
                    {
                        name = "phone",
                        // "recognized" is true when the phone is held or used
                        recognized = u.Status != PhoneStatus.NotInHand,
                        status = u.Status.ToString(),
                        similarity = (float?)null,
                        confidence = u.Phone.Confidence,
                        x = u.Phone.X,
                        y = u.Phone.Y,
                        width = u.Phone.Width,
                        height = u.Phone.Height
                    }).ToList()
                });

                frameNumber++;
            }

            object? snapshot = null;

            if (bestJpeg != null)
            {
                var saved = await _snapshots.SaveAsync(
                    bestJpeg, null, $"{Request.Scheme}://{Request.Host}", cancellationToken);

                snapshot = new
                {
                    fileName = saved!.FileName,
                    url = saved.Url,
                    timestamp = bestTime,
                    confidence = bestConfidence,
                    phoneCount = bestCount,
                    imageBase64 = Convert.ToBase64String(bestJpeg)
                };
            }

            return Ok(new
            {
                message = "Phone detection completed.",
                framesAnalysed = analysed,
                framesWithPhone,
                framesWithPhoneInHand,
                phoneSeconds = framesWithPhone * SampleEverySeconds,
                phoneInHandSeconds = framesWithPhoneInHand * SampleEverySeconds,
                maxPhonesInFrame = maxPhones,
                firstSeen = firstSeen < 0 ? (double?)null : firstSeen,
                lastSeen = lastSeen < 0 ? (double?)null : lastSeen,
                frames,
                snapshot
            });
        }
        catch (OperationCanceledException)
        {
            return StatusCode(499, "Video processing was cancelled.");
        }
        catch (InvalidOperationException ex)
        {
            return BadRequest(new { message = "Phone detection failed.", error = ex.Message });
        }
        finally
        {
            if (System.IO.File.Exists(tempFile))
                System.IO.File.Delete(tempFile);
        }
    }

    // ---------------------------------------------------------------
    // DETECT PHONES ON A LIVE STREAM (no file is saved here)
    // ---------------------------------------------------------------
    public record PhoneStreamRequest(string Url, bool IncludeFrame = true);

    [HttpPost("detect-stream")]
    public async Task<IActionResult> DetectStream([FromBody] PhoneStreamRequest request)
    {
        if (!Uri.TryCreate(request.Url, UriKind.Absolute, out var uri) ||
            (uri.Scheme != "http" && uri.Scheme != "https" && uri.Scheme != "rtsp"))
            return BadRequest("Only http, https and rtsp links are allowed.");

        var (jpeg, error) = await _streamHub.GetFrameAsync(request.Url);
        if (jpeg == null)
            return BadRequest(error);

        using var frame = Cv2.ImDecode(jpeg, ImreadModes.Color);
        if (frame.Empty())
            return BadRequest("Could not read a frame from the stream.");

        try
        {
            var analysis = _phoneUse.Analyze(frame);

            return Ok(new
            {
                totalPhones = analysis.Phones.Count,
                phonesInHand = analysis.Phones.Count(u => u.Status != PhoneStatus.NotInHand),
                detections = analysis.Phones.Select(u => new
                {
                    x = u.Phone.X,
                    y = u.Phone.Y,
                    width = u.Phone.Width,
                    height = u.Phone.Height,
                    confidence = u.Phone.Confidence,
                    status = u.Status.ToString(),
                    personIndex = u.PersonIndex
                }),
                frameBase64 = request.IncludeFrame && analysis.Phones.Count > 0 ? Convert.ToBase64String(jpeg) : null
            });
        }
        catch (InvalidOperationException ex)
        {
            return BadRequest(new { message = "Phone detection failed.", error = ex.Message });
        }
    }

    // ---------------------------------------------------------------
    // SAVE THE BEST LIVE SNAPSHOT (called once, when the stream is stopped)
    // ---------------------------------------------------------------
    public record SnapshotBox(int X, int Y, int Width, int Height, string? Label, string? Color);
    public record SaveSnapshotRequest(string ImageBase64, List<SnapshotBox>? Boxes);

    [HttpPost("save-snapshot")]
    public async Task<IActionResult> SaveSnapshot(
        [FromBody] SaveSnapshotRequest request, CancellationToken cancellationToken)
    {
        byte[] bytes;
        try { bytes = Convert.FromBase64String(request.ImageBase64); }
        catch (FormatException) { return BadRequest("Invalid image data."); }

        var saved = await _snapshots.SaveAsync(
            bytes, null, $"{Request.Scheme}://{Request.Host}", cancellationToken);

        return Ok(new { fileName = saved!.FileName, url = saved.Url });
    }
}