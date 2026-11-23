using System.Diagnostics;
using AIManagement.Api.services;
using AIManagement.Api.Services;
using Microsoft.AspNetCore.Mvc;
using OpenCvSharp;

namespace AIManagement.Api.Controllers;

[ApiController]
[Route("api/[controller]")]
public class CamerasController : ControllerBase
{
    private readonly StreamHub _streamHub;

    public CamerasController(StreamHub streamHub)
    {
        _streamHub = streamHub;
    }

    public record TestRequest(string Url);

    // Opens the stream, grabs one frame and reports what it found.
    // Always answers 200 with ok = true/false, so the page can show the reason.
    [HttpPost("test")]
    public async Task<IActionResult> Test([FromBody] TestRequest request)
    {
        if (!Uri.TryCreate(request.Url, UriKind.Absolute, out var uri) ||
            (uri.Scheme != "http" && uri.Scheme != "https" && uri.Scheme != "rtsp"))
            return Ok(new { ok = false, error = "Only http, https and rtsp links are allowed." });

        var timer = Stopwatch.StartNew();
        var (jpeg, error) = await _streamHub.GetFrameAsync(request.Url);

        if (jpeg == null)
            return Ok(new { ok = false, error = error ?? "No frame was received from the camera." });

        using var frame = Cv2.ImDecode(jpeg, ImreadModes.Color);
        if (frame.Empty())
            return Ok(new { ok = false, error = "The camera sent data that is not a picture." });

        // Small thumbnail for the camera card
        var scale = 480.0 / frame.Width;
        using var small = new Mat();
        Cv2.Resize(frame, small, new Size(480, (int)(frame.Height * scale)));

        return Ok(new
        {
            ok = true,
            width = frame.Width,
            height = frame.Height,
            ms = timer.ElapsedMilliseconds,
            preview = Convert.ToBase64String(small.ToBytes(".jpg")),
            error = (string?)null
        });
    }
}