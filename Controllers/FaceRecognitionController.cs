using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Xml.Linq;
using AIManagement.Api.DTO;
using AIManagement.Api.Models;
using AIManagement.Api.Repositories;
using AIManagement.Api.services;
using AIManagement.Api.Services;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using OpenCvSharp;


namespace AIManagement.Api.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class FaceRecognitionController : ControllerBase
    {
        private readonly FaceInferenceService _faceInferenceService;
        private readonly FaceRepository _faceRepository;
        private readonly RtspFaceRecognitionService _rtspService;
        private readonly LocalCameraTestService _localCameraTestService;
        private readonly StreamHub _streamHub;
        private readonly SnapshotService _snapshots;
        private readonly StickyIdentityService _sticky;

        public FaceRecognitionController(
            FaceInferenceService faceInferenceService,
            FaceRepository faceRepository,
            RtspFaceRecognitionService rtspService,
            LocalCameraTestService localCameraTestService,
            StreamHub streamHub,
            SnapshotService snapshots,
            StickyIdentityService sticky)
        {
            _faceInferenceService = faceInferenceService;
            _faceRepository = faceRepository;
            _rtspService = rtspService;
            _localCameraTestService = localCameraTestService;
            _streamHub = streamHub;
            _snapshots = snapshots;
            _sticky = sticky;
        }

        // -----------------------------------------------------------
        // Helpers used by video snapshot saving
        // -----------------------------------------------------------
        private sealed record PendingSnapshot(
            byte[] Frame, int X, int Y, int Width, int Height,
            float Confidence, float Score, string EmployeeId, string Name,
            float Similarity, float[] Embedding);

        private static float Cosine(float[] a, float[] b)
        {
            float dot = 0, na = 0, nb = 0;
            for (var i = 0; i < a.Length; i++)
            {
                dot += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }
            return dot / (MathF.Sqrt(na) * MathF.Sqrt(nb) + 1e-9f);
        }

        // ---------------------------------------------------------------
        // REGISTER FACE
        // ---------------------------------------------------------------
        [HttpPost("register")]
        [Consumes("multipart/form-data")]
        public async Task<IActionResult> Register(
            [FromForm] RegisterFaceRequest request,
            CancellationToken cancellationToken)
        {
            var emp_id = request.EmployeeId;
            var name = request.Name;
            var images = request.Images;

            if (string.IsNullOrWhiteSpace(emp_id))
                return BadRequest("Employee ID is required.");

            if (string.IsNullOrWhiteSpace(name))
                return BadRequest("Person name is required.");

            if (images == null || images.Count == 0)
                return BadRequest("Please upload at least one image.");

            if (images.Count > 5)
                return BadRequest("You can upload a maximum of 5 images at once.");

            foreach (var image in images)
            {
                if (image == null || image.Length == 0)
                    return BadRequest("One of the uploaded images is empty.");

                if (image.Length > 10 * 1024 * 1024)
                    return BadRequest("Each image must be smaller than 10 MB.");
            }

            var embeddings = new List<float[]>();

            try
            {
                foreach (var image in images)
                {
                    await using var stream = new MemoryStream();

                    await image.CopyToAsync(stream, cancellationToken);

                    var embedding =
                        await _faceInferenceService.GetEmbeddingAsync(
                            stream.ToArray(), cancellationToken);

                    embeddings.Add(embedding);
                }
            }
            catch (InvalidOperationException ex)
            {
                return BadRequest(new
                {
                    message = "Face registration failed. Check that every image contains a clear face.",
                    error = ex.Message
                });
            }

            foreach (var embedding in embeddings)
            {
                await _faceRepository.SaveEmbeddingAsync(emp_id, name, embedding);
            }

            return Ok(new
            {
                message = "Person registered successfully.",
                name = name.Trim(),
                totalImages = embeddings.Count,
                totalEmbeddingsSaved = embeddings.Count
            });
        }

        // ---------------------------------------------------------------
        // ADD ONE MORE PHOTO TO AN EXISTING EMPLOYEE (with same-person guard)
        // ---------------------------------------------------------------
        [HttpPost("people/{employeeId}/images")]
        [Consumes("multipart/form-data")]
        public async Task<IActionResult> AddImage(
            string employeeId,
            [FromForm] string name,
            IFormFile image,
            CancellationToken cancellationToken)
        {
            if (string.IsNullOrWhiteSpace(name))
                return BadRequest("Person name is required.");

            if (image == null || image.Length == 0)
                return BadRequest("Please upload an image.");

            if (image.Length > 10 * 1024 * 1024)
                return BadRequest("Image must be smaller than 10 MB.");

            try
            {
                await using var stream = new MemoryStream();
                await image.CopyToAsync(stream, cancellationToken);
                byte[] bytes = stream.ToArray();

                // 1. Exactly one face in the photo
                var faces = await _faceInferenceService.DetectFacesAsync(
                    bytes, cancellationToken);

                if (faces.Count != 1)
                    return BadRequest("The photo must contain exactly one face.");

                // 2. Create the embedding
                var embedding = await _faceInferenceService.GetEmbeddingAsync(
                    bytes, cancellationToken);

                // 3. Guard: must look like the person already registered
                var best = await _faceRepository.GetBestSimilarityForPersonAsync(
                    employeeId, embedding);

                if (best is null)
                    return NotFound("This employee has no registered faces yet.");

                if (best < 0.30f)
                    return BadRequest("This doesn't look like the registered person.");

                // 4. Save
                await _faceRepository.SaveEmbeddingAsync(employeeId, name, embedding);

                return Ok(new
                {
                    message = "Photo added.",
                    similarityToExisting = best
                });
            }
            catch (InvalidOperationException ex)
            {
                return BadRequest(new
                {
                    message = "Could not add this photo.",
                    error = ex.Message
                });
            }
        }

        // ---------------------------------------------------------------
        // ROSTER
        // ---------------------------------------------------------------
        [HttpGet("people")]
        public async Task<IActionResult> GetPeople()
        {
            return Ok(await _faceRepository.GetPeopleDetailsAsync());
        }

        [HttpDelete("people/{employeeId}")]
        public async Task<IActionResult> DeletePerson(string employeeId)
        {
            var deleted = await _faceRepository.DeletePersonAsync(employeeId);
            return deleted ? NoContent() : NotFound();
        }

        // ---------------------------------------------------------------
        // RECOGNIZE FROM IMAGE URL
        // ---------------------------------------------------------------
        public record UrlRequest(string Url);

        private static readonly HttpClient _http = new() { Timeout = TimeSpan.FromSeconds(15) };

        [HttpPost("recognize-url")]
        public async Task<IActionResult> RecognizeUrl(
            [FromBody] UrlRequest request,
            CancellationToken ct)
        {
            if (!Uri.TryCreate(request.Url, UriKind.Absolute, out var uri) ||
                (uri.Scheme != "http" && uri.Scheme != "https"))
                return BadRequest("Only http and https image links are allowed.");

            // Don't let the API be used to reach internal addresses
            try
            {
                var addresses = await Dns.GetHostAddressesAsync(uri.Host, ct);
                if (addresses.Any(a => IPAddress.IsLoopback(a) || IsPrivate(a)))
                    return BadRequest("This address is not allowed.");
            }
            catch
            {
                return BadRequest("Could not resolve the host of this link.");
            }

            byte[] bytes;
            try
            {
                using var response = await _http.GetAsync(
                    uri, HttpCompletionOption.ResponseHeadersRead, ct);

                if (!response.IsSuccessStatusCode)
                    return BadRequest($"The link returned status {(int)response.StatusCode}.");

                var type = response.Content.Headers.ContentType?.MediaType ?? "";
                if (!type.StartsWith("image/"))
                    return BadRequest("The link is not an image.");

                if (response.Content.Headers.ContentLength > 10 * 1024 * 1024)
                    return BadRequest("Image must be smaller than 10 MB.");

                bytes = await response.Content.ReadAsByteArrayAsync(ct);
            }
            catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
            {
                return BadRequest("Could not download the image from this link.");
            }

            if (bytes.Length == 0 || bytes.Length > 10 * 1024 * 1024)
                return BadRequest("Image must be between 1 byte and 10 MB.");

            // Reuse the existing recognize action
            using var ms = new MemoryStream(bytes);
            var file = new FormFile(ms, 0, bytes.Length, "image", "url-image.jpg")
            {
                Headers = new HeaderDictionary(),
                ContentType = "image/jpeg"
            };

            HttpContext.Items["sightingSource"] = "url:" + uri.Host;
            return await Recognize(file, ct);
        }

        private static bool IsPrivate(IPAddress ip)
        {
            if (ip.IsIPv6LinkLocal || ip.IsIPv6SiteLocal) return true;
            var b = ip.GetAddressBytes();
            if (b.Length != 4) return false;
            return b[0] == 10 || b[0] == 0
                || (b[0] == 172 && b[1] >= 16 && b[1] <= 31)
                || (b[0] == 192 && b[1] == 168)
                || (b[0] == 169 && b[1] == 254);
        }

        // ---------------------------------------------------------------
        // RECOGNIZE FACE (image upload; also used by url and stream)
        // ---------------------------------------------------------------
        [HttpPost("recognize")]
        [Consumes("multipart/form-data")]
        public async Task<IActionResult> Recognize(
            IFormFile image,
            CancellationToken cancellationToken,
            [FromQuery] bool clean = false)
        {
            if (image == null || image.Length == 0)
                return BadRequest("Please upload an image.");

            if (image.Length > 10 * 1024 * 1024)
                return BadRequest("Image must be smaller than 10 MB.");

            try
            {
                await using var stream = new MemoryStream();

                await image.CopyToAsync(stream, cancellationToken);

                byte[] imageBytes = stream.ToArray();

                var faces = await _faceInferenceService.AnalyzeFacesAsync(
                    imageBytes, cancellationToken);

                var detections = new List<RecognitionResult>();
                var annotations = new List<FaceAnnotation>();

                // Where this image came from, and how snapshots should be saved
                var source = HttpContext.Items["sightingSource"] as string ?? "upload-image";
                var mode = HttpContext.Items["sightingMode"] is SnapshotMode snapshotMode
                    ? snapshotMode
                    : SnapshotMode.EveryFace;

                const float threshold = 0.40f;

                // Match every face first, so names can be made sticky before labels are built
                var matches = new List<(string Name, string EmployeeId, float Similarity)?>();
                foreach (var candidate in faces)
                    matches.Add(await _faceRepository.FindMatchAsync(candidate.Embedding));

                // Live streams only: a recognized name stays until that face leaves the frame
                var stickyKey = HttpContext.Items["stickyKey"] as string;
                IReadOnlyList<StickyResult>? sticky = null;

                if (stickyKey != null)
                {
                    sticky = _sticky.Resolve(stickyKey, faces.Select((candidate, n) =>
                    {
                        var found = matches[n];
                        var known = found != null && found.Value.Similarity >= threshold;

                        return new StickyFace(
                            candidate.X, candidate.Y, candidate.Width, candidate.Height,
                            candidate.Embedding,
                            known,
                            known ? found!.Value.EmployeeId : "UNKNOWN",
                            known ? found!.Value.Name : "Unknown",
                            found?.Similarity ?? 0f);
                    }).ToList());
                }

                foreach (var (face, index) in faces.Select((f, n) => (f, n)))
                {
                    var match = matches[index];

                    bool recognized =
                        match != null &&
                        match.Value.Similarity >= threshold;

                    string name = recognized
                        ? match!.Value.Name
                        : "Unknown";

                    string? employeeId = recognized
                        ? match!.Value.EmployeeId
                        : null;

                    float? similarity = match?.Similarity;

                    // Sticky name: a face that was recognized earlier keeps its identity
                    if (sticky != null && sticky[index].Recognized)
                    {
                        var resolved = sticky[index];
                        recognized = true;
                        name = resolved.Name;
                        employeeId = resolved.EmployeeId;
                        similarity = resolved.Similarity;
                    }

                    detections.Add(new RecognitionResult
                    {
                        Name = name,
                        Recognized = recognized,
                        EmployeeId = employeeId,
                        Similarity = similarity,
                        DetectionConfidence = face.DetectionConfidence,
                        X = face.X,
                        Y = face.Y,
                        Width = face.Width,
                        Height = face.Height
                    });

                    var snapshotId = recognized ? employeeId! : "UNKNOWN";

                    try
                    {
                        await _snapshots.OfferAsync(
                            imageBytes, face.X, face.Y, face.Width, face.Height,
                            face.DetectionConfidence, face.Embedding,
                            snapshotId, name, similarity ?? 0f, source, mode);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Snapshot save failed: {ex.Message}");
                    }

                    annotations.Add(new FaceAnnotation(
                        face.X,
                        face.Y,
                        face.Width,
                        face.Height,
                        name,
                        similarity ?? 0
                    ));
                }

                byte[] annotatedImage = clean
                    ? imageBytes
                    : _faceInferenceService.CreateAnnotatedImage(imageBytes, annotations);

                return Ok(new
                {
                    message = "Face recognition completed.",
                    totalFaces = detections.Count,
                    detections,
                    imageBase64 = Convert.ToBase64String(annotatedImage)
                });
            }
            catch (InvalidOperationException ex)
            {
                return BadRequest(new
                {
                    message = "Face recognition failed.",
                    error = ex.Message
                });
            }
        }

        // ---------------------------------------------------------------
        // SIGHTINGS (saved snapshots)
        // ---------------------------------------------------------------
        [HttpGet("sightings")]
        public async Task<IActionResult> GetSightings([FromQuery] int limit = 100)
            => Ok(await _faceRepository.GetSightingsAsync(Math.Clamp(limit, 1, 500)));

        [HttpGet("sightings/{id:int}/image")]
        public async Task<IActionResult> GetSightingImage(int id)
        {
            var relative = await _faceRepository.GetSightingPathAsync(id);
            if (relative == null) return NotFound();

            var full = _snapshots.Resolve(relative);
            if (full == null || !System.IO.File.Exists(full)) return NotFound();

            return PhysicalFile(full, "image/jpeg");
        }

        // ---------------------------------------------------------------
        // TEMPORARY LOCAL WEBCAM TEST
        // ---------------------------------------------------------------
        [HttpPost("start-local-camera")]
        public async Task<IActionResult> StartLocalCamera()
        {
            try
            {
                await _localCameraTestService.StartAsync();

                return Ok(new
                {
                    message = "Local webcam started successfully.",
                    status = "Running"
                });
            }
            catch (InvalidOperationException ex)
            {
                return BadRequest(new
                {
                    message = ex.Message
                });
            }
        }

        [HttpPost("stop-local-camera")]
        public async Task<IActionResult> StopLocalCamera()
        {
            await _localCameraTestService.StopAsync();

            return Ok(new
            {
                message = "Local webcam stopped.",
                status = "Stopped"
            });
        }

        [HttpGet("local-camera-status")]
        public IActionResult LocalCameraStatus()
        {
            return Ok(new
            {
                isRunning = _localCameraTestService.IsRunning
            });
        }

        [HttpGet("local-camera-frame")]
        public IActionResult GetLocalCameraFrame()
        {
            var frame = _localCameraTestService.GetLatestFrame();

            if (frame == null)
            {
                return NotFound(new
                {
                    message = "No processed camera frame is available yet."
                });
            }

            return File(frame, "image/jpeg");
        }

        // ---------------------------------------------------------------
        // RECOGNIZE VIDEO
        // ---------------------------------------------------------------
        [HttpPost("recognize-video")]
        [Consumes("multipart/form-data")]
        [RequestSizeLimit(100 * 1024 * 1024)]
        [RequestFormLimits(MultipartBodyLengthLimit = 100 * 1024 * 1024)]
        public async Task<IActionResult> RecognizeVideo(
            IFormFile video,
            CancellationToken cancellationToken)
        {
            if (video == null || video.Length == 0)
                return BadRequest("Please upload a video.");

            if (video.Length > 100 * 1024 * 1024)
                return BadRequest("Video must be smaller than 100 MB.");

            var allowedExtensions = new[]
            {
                ".mp4", ".avi", ".mov", ".mkv", ".webm"
            };

            var extension = Path.GetExtension(video.FileName).ToLowerInvariant();

            if (!allowedExtensions.Contains(extension))
                return BadRequest("Unsupported video format.");

            var tempFile = Path.Combine(
                Path.GetTempPath(),
                $"{Guid.NewGuid()}{extension}");

            try
            {
                await using (var fileStream = new FileStream(
                    tempFile, FileMode.Create, FileAccess.Write))
                {
                    await video.CopyToAsync(fileStream, cancellationToken);
                }

                using var capture = new VideoCapture(tempFile);

                if (!capture.IsOpened())
                    return BadRequest("Unable to open the uploaded video.");

                double fps = capture.Fps;

                if (fps <= 0 || double.IsNaN(fps) || double.IsInfinity(fps))
                    fps = 25;

                // How often a frame is analysed. 1.0 halves the work, but the boxes
                // on the video then update only once per second.
                const double SampleEverySeconds = 0.5;
                double frameInterval = Math.Max(1, fps * SampleEverySeconds);

                using var frame = new Mat();

                long frameNumber = 0;

                var recognizedPeople = new Dictionary<string, dynamic>();

                // Light per-sample data the browser uses to draw boxes over the video.
                var frames = new List<object>();

                // "Best Snapshot" card: the frame with the most people in it.
                // If two frames have the same number of people, the one with the
                // higher detection confidence (among recognized faces) wins.
                byte[]? topFrameBytes = null;
                List<FaceAnnotation>? topAnnotations = null;
                int topFaceCount = 0;
                int topRecognizedCount = 0;
                float topConfidence = 0;
                double topTimestamp = 0;

                // Follows every face across the sampled frames.
                var tracker = new FaceTracker(maxGapSeconds: 3.0);

                while (true)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    // Frames we skip only need to be decoded, not converted to an image
                    if (frameNumber % (long)frameInterval != 0)
                    {
                        if (!capture.Grab())
                            break;

                        frameNumber++;
                        continue;
                    }

                    if (!capture.Read(frame) || frame.Empty())
                        break;

                    double timestamp = frameNumber / fps;

                    Cv2.ImEncode(".jpg", frame, out byte[] imageBytes);

                    var faces = await _faceInferenceService.AnalyzeFacesAsync(
                        imageBytes, cancellationToken);

                    // Link this frame's faces to tracks (by appearance and position).
                    var observations = faces
                        .Select(f => new FaceObservation(f.X, f.Y, f.Width, f.Height, f.Embedding))
                        .ToList();

                    var tracks = tracker.Associate(observations, timestamp);

                    var annotations = new List<FaceAnnotation>();
                    var frameDetections = new List<object>();

                    // Recognized faces in this frame, and the best detection confidence among them.
                    int recognizedCount = 0;
                    float frameBest = 0;

                    for (var i = 0; i < faces.Count; i++)
                    {
                        var face = faces[i];
                        var track = tracks[i];

                        var match = await _faceRepository.FindMatchAsync(
                            face.Embedding);

                        const float threshold = 0.40f;

                        bool isRecognized =
                            match != null &&
                            match.Value.Similarity >= threshold;

                        string name = isRecognized
                            ? match!.Value.Name
                            : "Unknown";

                        float? similarity = match?.Similarity;

                        annotations.Add(new FaceAnnotation(
                            face.X,
                            face.Y,
                            face.Width,
                            face.Height,
                            name,
                            similarity ?? 0));

                        frameDetections.Add(new
                        {
                            name,
                            recognized = isRecognized,
                            similarity,
                            confidence = face.DetectionConfidence,
                            x = face.X,
                            y = face.Y,
                            width = face.Width,
                            height = face.Height
                        });

                        if (isRecognized)
                        {
                            recognizedCount++;

                            if (face.DetectionConfidence > frameBest)
                                frameBest = face.DetectionConfidence;

                            // This track was this person in this frame
                            track.Vote(match!.Value.EmployeeId, name, similarity ?? 0f);

                            if (!recognizedPeople.ContainsKey(name))
                            {
                                recognizedPeople[name] = new
                                {
                                    name,
                                    firstSeen = timestamp,
                                    lastSeen = timestamp,
                                    matchedFrames = 1,
                                    bestSimilarity = similarity
                                };
                            }
                            else
                            {
                                dynamic previous = recognizedPeople[name];

                                recognizedPeople[name] = new
                                {
                                    name,
                                    firstSeen = (double)previous.firstSeen,
                                    lastSeen = timestamp,
                                    matchedFrames = (int)previous.matchedFrames + 1,
                                    bestSimilarity = Math.Max(
                                        (float)previous.bestSimilarity,
                                        similarity!.Value)
                                };
                            }
                        }

                        // Quality of this view of the face. The track keeps only its best one.
                        var quality = FaceQuality.Score(
                            frame, face.X, face.Y, face.Width, face.Height,
                            face.DetectionConfidence);

                        track.Offer(
                            quality, imageBytes, observations[i],
                            face.DetectionConfidence, similarity ?? 0f);
                    }

                    // Every sample is recorded (even with no faces) so the overlay
                    // clears its boxes when nobody is on screen.
                    frames.Add(new
                    {
                        timestamp,
                        detections = frameDetections
                    });

                    // Card frame: needs at least one recognized person, then most people wins,
                    // and the old confidence rule breaks ties.
                    if (recognizedCount > 0 &&
                        (faces.Count > topFaceCount ||
                         (faces.Count == topFaceCount && frameBest > topConfidence)))
                    {
                        topFaceCount = faces.Count;
                        topRecognizedCount = recognizedCount;
                        topConfidence = frameBest;
                        topFrameBytes = imageBytes;
                        topAnnotations = annotations;
                        topTimestamp = timestamp;
                    }

                    frameNumber++;
                }

                // Annotate once, at the end, for the chosen frame.
                object? snapshot = null;

                if (topFrameBytes != null && topAnnotations != null)
                {
                    byte[] annotatedImage =
                        _faceInferenceService.CreateAnnotatedImage(
                            topFrameBytes, topAnnotations);

                    snapshot = new
                    {
                        timestamp = topTimestamp,
                        confidence = topConfidence,
                        faceCount = topFaceCount,
                        recognizedCount = topRecognizedCount,
                        imageBase64 = Convert.ToBase64String(annotatedImage)
                    };
                }

                // Save once per person (tracks of the same person are merged), plus one
                // per distinct unknown face that was visible for at least 2 samples.
                foreach (var s in tracker.SelectSnapshots(minUnknownFrames: 2, maxUnknown: 50))
                {
                    try
                    {
                        await _snapshots.OfferAsync(
                            s.Frame, s.X, s.Y, s.Width, s.Height,
                            s.Confidence, s.Embedding,
                            s.EmployeeId, s.Name, s.Similarity,
                            "upload-video", SnapshotMode.EveryFace);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Snapshot save failed: {ex.Message}");
                    }
                }

                return Ok(new
                {
                    message = "Video recognition completed.",
                    totalRecognizedPeople = recognizedPeople.Count,
                    recognizedPeople = recognizedPeople.Values,
                    frames,
                    snapshot
                });
            }
            catch (OperationCanceledException)
            {
                return StatusCode(499, "Video processing was cancelled.");
            }
            finally
            {
                if (System.IO.File.Exists(tempFile))
                    System.IO.File.Delete(tempFile);
            }
        }

        public class RecognizeDetectedRequest
        {
            public IFormFile Image { get; set; } = null!;

            public float X { get; set; }
            public float Y { get; set; }
            public float Width { get; set; }
            public float Height { get; set; }

            public float Confidence { get; set; }

            public float LeftEyeX { get; set; }
            public float LeftEyeY { get; set; }

            public float RightEyeX { get; set; }
            public float RightEyeY { get; set; }

            public float NoseX { get; set; }
            public float NoseY { get; set; }

            public float LeftMouthX { get; set; }
            public float LeftMouthY { get; set; }

            public float RightMouthX { get; set; }
            public float RightMouthY { get; set; }
        }

        // ---------------------------------------------------------------
        // RTSP (service based)
        // ---------------------------------------------------------------
        public class StartRtspRequest
        {
            public string CameraId { get; set; } = "";
            public string RtspUrl { get; set; } = "";
        }

        [HttpPost("rtsp/start")]
        public IActionResult StartRtspCamera(
            [FromBody] StartRtspRequest request)
        {
            if (request == null ||
                string.IsNullOrWhiteSpace(request.CameraId) ||
                string.IsNullOrWhiteSpace(request.RtspUrl))
            {
                return BadRequest("CameraId and RtspUrl are required.");
            }

            if (!Uri.TryCreate(request.RtspUrl, UriKind.Absolute, out var uri) ||
                (uri.Scheme != "rtsp" && uri.Scheme != "rtsps"))
            {
                return BadRequest("Please provide a valid RTSP URL.");
            }

            bool started = _rtspService.StartCamera(
                request.CameraId,
                request.RtspUrl);

            if (!started)
                return Conflict("Camera is already running.");

            return Ok(new
            {
                message = "RTSP camera processing started.",
                cameraId = request.CameraId,
                rtspUrl = request.RtspUrl
            });
        }

        [HttpPost("rtsp/stop/{cameraId}")]
        public async Task<IActionResult> StopRtspCamera(string cameraId)
        {
            bool stopped = await _rtspService.StopCameraAsync(cameraId);

            if (!stopped)
                return NotFound("Camera not found.");

            return Ok(new
            {
                message = "RTSP camera stopped.",
                cameraId
            });
        }

        [HttpGet("rtsp/status/{cameraId}")]
        public IActionResult GetRtspStatus(string cameraId)
        {
            var status = _rtspService.GetCameraStatus(cameraId);

            if (status == null)
                return NotFound("Camera not found.");

            return Ok(status);
        }

        [HttpGet("rtsp/results/{cameraId}")]
        public IActionResult GetRtspResults(string cameraId)
        {
            var results = _rtspService.GetCameraResults(cameraId);

            if (results == null)
                return NotFound("Camera not found.");

            return Ok(results);
        }

        // ---------------------------------------------------------------
        // LIVE STREAM (phone / IP camera link)
        // ---------------------------------------------------------------
        [HttpGet("stream/mjpeg")]
        public async Task StreamMjpeg([FromQuery] string url, CancellationToken ct)
        {
            if (!Uri.TryCreate(url, UriKind.Absolute, out var uri) ||
                (uri.Scheme != "http" && uri.Scheme != "https" && uri.Scheme != "rtsp"))
            {
                Response.StatusCode = 400;
                return;
            }

            try
            {
                var started = false;

                while (!ct.IsCancellationRequested)
                {
                    var (jpeg, _) = await _streamHub.GetFrameAsync(url);

                    if (jpeg == null)
                    {
                        // Camera offline: answer with a status before any video was sent
                        if (!started) Response.StatusCode = 503;
                        break;
                    }

                    if (!started)
                    {
                        Response.ContentType = "multipart/x-mixed-replace; boundary=frame";
                        started = true;
                    }

                    var header = Encoding.ASCII.GetBytes(
                        $"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {jpeg.Length}\r\n\r\n");

                    await Response.Body.WriteAsync(header, ct);
                    await Response.Body.WriteAsync(jpeg, ct);
                    await Response.Body.WriteAsync("\r\n"u8.ToArray(), ct);
                    await Response.Body.FlushAsync(ct);

                    await Task.Delay(33, ct); // about 30 fps
                }
            }
            catch (OperationCanceledException)
            {
                // The page closed the video. This is normal, not an error.
            }
            catch (IOException)
            {
                // The connection dropped while sending. Also normal.
            }
        }

        public record StreamRequest(string Url);

        [HttpPost("recognize-stream")]
        public async Task<IActionResult> RecognizeStream([FromBody] StreamRequest request)
        {
            // Only allow stream-type links
            if (!Uri.TryCreate(request.Url, UriKind.Absolute, out var uri) ||
                (uri.Scheme != "http" && uri.Scheme != "https" && uri.Scheme != "rtsp"))
                return BadRequest("Only http, https and rtsp links are allowed.");

            // Latest frame from the already-open stream (opened once, then reused)
            var (jpeg, error) = await _streamHub.GetFrameAsync(request.Url);
            if (jpeg == null)
                return BadRequest(error);

            // Wrap the frame as an uploaded file and reuse the existing recognize action
            using var ms = new MemoryStream(jpeg);
            var file = new FormFile(ms, 0, jpeg.Length, "image", "frame.jpg")
            {
                Headers = new HeaderDictionary(),
                ContentType = "image/jpeg"
            };

            HttpContext.Items["sightingSource"] = "stream:" + uri.Host;
            HttpContext.Items["sightingMode"] = SnapshotMode.BestPerSession;
            HttpContext.Items["stickyKey"] = request.Url;

            return await Recognize(file, HttpContext.RequestAborted, clean: true);
        }

        // ---------------------------------------------------------------
        // DETECT FACES ONLY - YOLO
        // ---------------------------------------------------------------
        [HttpPost("detect")]
        [Consumes("multipart/form-data")]
        public async Task<IActionResult> DetectFaces(
            IFormFile image,
            CancellationToken cancellationToken)
        {
            if (image == null || image.Length == 0)
                return BadRequest("Please upload an image.");

            if (image.Length > 10 * 1024 * 1024)
                return BadRequest("Image must be smaller than 10 MB.");

            try
            {
                await using var stream = new MemoryStream();

                await image.CopyToAsync(stream, cancellationToken);

                byte[] imageBytes = stream.ToArray();

                // YOLO only. No ArcFace and no database query.
                var faces = await _faceInferenceService.DetectFacesAsync(
                    imageBytes, cancellationToken);

                var detections = faces.Select(face => new FaceDetectionResponse
                {
                    X = face.Box.X,
                    Y = face.Box.Y,
                    Width = face.Box.Width,
                    Height = face.Box.Height,
                    Confidence = face.Confidence,

                    Landmarks = face.Landmarks
                        .Select(point => new[] { point.X, point.Y })
                        .ToArray()
                }).ToList();

                return Ok(new
                {
                    message = "Face detection completed.",
                    totalFaces = detections.Count,
                    detections
                });
            }
            catch (InvalidOperationException ex)
            {
                return BadRequest(new
                {
                    message = "Face detection failed.",
                    error = ex.Message
                });
            }
        }

        // ---------------------------------------------------------------
        // RECOGNIZE AN ALREADY DETECTED FACE - ARCFACE ONLY
        // ---------------------------------------------------------------
        [HttpPost("recognize-detected")]
        [Consumes("multipart/form-data")]
        public async Task<IActionResult> RecognizeDetected(
            [FromForm] RecognizeDetectedRequest request,
            CancellationToken cancellationToken)
        {
            if (request.Image == null || request.Image.Length == 0)
                return BadRequest("Please upload an image.");

            if (request.Image.Length > 10 * 1024 * 1024)
                return BadRequest("Image must be smaller than 10 MB.");

            // Validate face coordinates and landmarks.
            var values = new[]
            {
                request.X, request.Y,
                request.Width, request.Height,
                request.Confidence,
                request.LeftEyeX, request.LeftEyeY,
                request.RightEyeX, request.RightEyeY,
                request.NoseX, request.NoseY,
                request.LeftMouthX, request.LeftMouthY,
                request.RightMouthX, request.RightMouthY
            };

            if (values.Any(value => !float.IsFinite(value)))
                return BadRequest("Invalid face coordinates or landmarks.");

            if (request.Width <= 0 || request.Height <= 0)
                return BadRequest("Invalid face dimensions.");

            try
            {
                await using var stream = new MemoryStream();

                await request.Image.CopyToAsync(stream, cancellationToken);

                byte[] imageBytes = stream.ToArray();

                // Original landmark order returned by YOLO:
                // Left eye, right eye, nose, left mouth, right mouth.
                Point2f[] landmarks =
                {
                    new(request.LeftEyeX, request.LeftEyeY),
                    new(request.RightEyeX, request.RightEyeY),
                    new(request.NoseX, request.NoseY),
                    new(request.LeftMouthX, request.LeftMouthY),
                    new(request.RightMouthX, request.RightMouthY)
                };

                // ArcFace only. YOLO is NOT executed again.
                var embedding = await _faceInferenceService.CreateEmbeddingAsync(
                    imageBytes,
                    landmarks,
                    cancellationToken);

                // Find matching person in PostgreSQL.
                var match = await _faceRepository.FindMatchAsync(embedding);

                const float threshold = 0.40f;

                bool recognized =
                    match != null &&
                    match.Value.Similarity >= threshold;

                string name = recognized
                    ? match!.Value.Name
                    : "Unknown";

                float? similarity = match?.Similarity;

                var annotation = new FaceAnnotation(
                    (int)request.X,
                    (int)request.Y,
                    (int)request.Width,
                    (int)request.Height,
                    name,
                    similarity ?? 0
                );

                byte[] annotatedImage =
                    _faceInferenceService.CreateAnnotatedImage(
                        imageBytes,
                        new[] { annotation });

                return Ok(new
                {
                    message = "Face recognition completed.",
                    name,
                    recognized,
                    similarity,
                    detectionConfidence = request.Confidence,
                    x = request.X,
                    y = request.Y,
                    width = request.Width,
                    height = request.Height,
                    imageBase64 = Convert.ToBase64String(annotatedImage)
                });
            }
            catch (InvalidOperationException ex)
            {
                return BadRequest(new
                {
                    message = "Face recognition failed.",
                    error = ex.Message
                });
            }
        }
    }
}