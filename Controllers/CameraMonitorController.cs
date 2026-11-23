using AIManagement.Api.services;
using Microsoft.AspNetCore.Mvc;
using OpenCvSharp;

namespace AIManagement.Api.Controllers;

[ApiController]
[Route("api/[controller]")]
public class CameraMonitorController : ControllerBase
{
    public record CameraDto(string Id, string Name, string Location, string Url, List<string>? Modules);

    private readonly CameraStore _store;
    private readonly CameraMonitor _monitor;
    private readonly StreamHub _hub;


    public CameraMonitorController(CameraStore store, CameraMonitor monitor, StreamHub hub)
    {
        _store = store;
        _monitor = monitor;
        _hub = hub;
    }

    // Temporary: the Cameras page pushes its whole list here whenever it changes
    [HttpPost("sync")]
    public IActionResult Sync([FromBody] List<CameraDto> cameras)
    {
        const string tenantId = "default";   // later: from the JWT

        var records = cameras
            .Where(c => Uri.TryCreate(c.Url, UriKind.Absolute, out var u) &&
                        (u.Scheme == "http" || u.Scheme == "https" || u.Scheme == "rtsp"))
            .Select(c => new CameraRecord
            {
                Id = c.Id,
                TenantId = tenantId,
                Name = c.Name,
                Location = c.Location,
                Url = c.Url,
                Modules = c.Modules ?? new()
            })
            .ToList();

        _store.ReplaceForTenant(tenantId, records);
        return Ok(new { saved = records.Count });
    }

    // Open this in the browser to see what the worker is doing
    [HttpGet("status")]
    public IActionResult Status() => Ok(_monitor.States.Values);


    [HttpGet("{id}/state")]
    public IActionResult State(string id) =>
    _monitor.States.TryGetValue(id, out var s) ? Ok(s) : NotFound();

    [HttpGet("{id}/best")]
    public IActionResult Best(string id) =>
        _monitor.States.TryGetValue(id, out var s) && s.BestJpeg != null
            ? File(s.BestJpeg, "image/jpeg") : NotFound();

    // Smooth video: fresh frames from StreamHub, with the worker's latest boxes drawn on top
    [HttpGet("{id}/video")]
    public async Task Video(string id, CancellationToken ct)
    {
        var cam = _store.All().FirstOrDefault(c => c.Id == id);
        if (cam == null) { Response.StatusCode = 404; return; }

        Response.ContentType = "multipart/x-mixed-replace; boundary=frame";
        try
        {
            while (!ct.IsCancellationRequested)
            {
                var (jpeg, _) = await _hub.GetFrameAsync(cam.Url);
                if (jpeg != null)
                {
                    var overlay = _monitor.States.TryGetValue(id, out var s) ? s.Overlay : null;
                    if (overlay != null && (DateTime.UtcNow - overlay.At).TotalSeconds <= 1.5)
                    {
                        using var frame = Cv2.ImDecode(jpeg, ImreadModes.Color);
                        OverlayPainter.Draw(frame, overlay);
                        jpeg = frame.ToBytes(".jpg");
                    }

                    await Response.WriteAsync(
                        $"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {jpeg.Length}\r\n\r\n", ct);
                    await Response.Body.WriteAsync(jpeg, ct);
                    await Response.WriteAsync("\r\n", ct);
                    await Response.Body.FlushAsync(ct);
                }
                await Task.Delay(80, ct);
            }
        }
        catch (OperationCanceledException) { }
    }
}