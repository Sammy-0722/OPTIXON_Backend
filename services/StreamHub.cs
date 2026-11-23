using OpenCvSharp;

namespace AIManagement.Api.services;

public sealed class FrameReader
{
    private readonly string _url;
    private readonly Action<FrameReader> _onStopped;
    private readonly object _lock = new();
    private Mat? _latest;
    private byte[]? _jpeg;
    private long _version;
    private long _jpegVersion = -1;
    private long _lastRequestTicks = Environment.TickCount64;
    private readonly long _createdTicks = Environment.TickCount64;
    private long _lastFrameTicks; // 0 until the first frame arrives
    private volatile bool _stop;

    public string Url => _url;

    // A frame arrived in the last 4 seconds
    private bool FrameFresh =>
        _lastFrameTicks != 0 && Environment.TickCount64 - Interlocked.Read(ref _lastFrameTicks) < 4000;

    // Working: no error, and frames are still arriving
    public bool IsHealthy => Error == null && FrameFresh;

    // Still connecting and waiting for its first picture. OpenCV itself gives up after about
    // 30 seconds, so a new reader is never started before then (it would only add one more
    // blocked connection attempt on top of the one already waiting).
    public bool IsOpening =>
        Error == null && _lastFrameTicks == 0 && Environment.TickCount64 - _createdTicks < 40000;
    public string? Error { get; private set; }

    public FrameReader(string url, Action<FrameReader> onStopped)
    {
        _url = url;
        _onStopped = onStopped;
        new Thread(Run) { IsBackground = true }.Start();
    }

    private void Run()
    {
        try
        {
            using var cap = new VideoCapture(_url);
            if (!cap.IsOpened())
            {
                Error = "Could not open the stream. Check that the camera is on, on the same network, and the link is correct.";
                return;
            }

            using var frame = new Mat();

            // Keep reading until nobody has asked for a frame for 10 seconds
            while (!_stop &&
                   Environment.TickCount64 - Interlocked.Read(ref _lastRequestTicks) < 10_000)
            {
                if (!cap.Read(frame) || frame.Empty())
                {
                    Error = "The stream ended.";
                    return;
                }

                lock (_lock)
                {
                    // Reuse one buffer instead of allocating a new picture 25-30 times a second
                    _latest ??= new Mat();
                    frame.CopyTo(_latest);
                    _version++;
                    Interlocked.Exchange(ref _lastFrameTicks, Environment.TickCount64);
                }
            }
        }
        catch (Exception ex)
        {
            Error = $"Stream error: {ex.Message}";
        }
        finally
        {
            lock (_lock)
            {
                _latest?.Dispose();
                _latest = null;
                _jpeg = null;
            }
            _onStopped(this);
        }
    }

    public byte[]? GetJpeg()
    {
        Interlocked.Exchange(ref _lastRequestTicks, Environment.TickCount64);

        lock (_lock)
        {
            // A picture older than 4 seconds means the camera stopped sending, so don't show it as live
            if (_latest == null || !FrameFresh) return null;

            // Encode once per frame, however many callers ask for it
            if (_jpeg == null || _jpegVersion != _version)
            {
                _jpeg = _latest.ToBytes(".jpg");
                _jpegVersion = _version;
            }

            return _jpeg;
        }
    }

    public void Stop() => _stop = true;
}

public sealed class StreamHub
{
    // An offline camera is tried 3 times in total: right away, after 10 s, after 30 s.
    // If all three fail, the system stops asking until a person presses Test Connection.
    private static readonly TimeSpan[] Pauses =
    {
        TimeSpan.FromSeconds(10),
        TimeSpan.FromSeconds(30),
    };

    private const string OfflineMessage =
        "The camera is offline or not reachable, so the system stopped asking for it. " +
        "Check its power, Wi-Fi and link, then press Test Connection.";

    private readonly object _gate = new();
    private readonly Dictionary<string, FrameReader> _readers = new();
    private readonly Dictionary<string, DateTime> _pausedUntil = new();
    private readonly Dictionary<string, int> _strikes = new();

    // force = true is for a person pressing a button (Test Connection, opening a live view).
    // It clears the pause and tries right away. Polling loops never pass it.
    public async Task<(byte[]? Jpeg, string? Error)> GetFrameAsync(string url, bool force = false)
    {
        FrameReader reader;

        // One lock, so two requests for the same link can never open two readers
        lock (_gate)
        {
            if (force)
            {
                _pausedUntil.Remove(url);
                _strikes.Remove(url);
            }
            else if (_pausedUntil.TryGetValue(url, out var until))
            {
                if (DateTime.UtcNow < until) return (null, OfflineMessage);
                _pausedUntil.Remove(url);
            }

            // A reader that stopped getting frames (camera switched off, Wi-Fi dropped) is replaced
            // by a new one, so a camera that was restarted can connect again straight away
            if (_readers.TryGetValue(url, out reader!) && !reader.IsHealthy && !reader.IsOpening)
            {
                reader.Stop();
                _readers.Remove(url);
                reader = null!;
            }

            if (reader == null)
            {
                reader = new FrameReader(url, OnStopped);
                _readers[url] = reader;
            }
        }

        // Wait up to 5 seconds for the first frame only
        for (var i = 0; i < 50; i++)
        {
            var jpeg = reader.GetJpeg();
            if (jpeg != null)
            {
                lock (_gate) _strikes.Remove(url); // it works, so forget earlier failures
                return (jpeg, null);
            }

            if (reader.Error != null) return (null, reader.Error);
            await Task.Delay(100);
        }

        return (null, "The camera did not answer in time.");
    }

    private void OnStopped(FrameReader reader)
    {
        lock (_gate)
        {
            // A reader that was already replaced must not touch the new one or count as a failure
            if (!_readers.TryGetValue(reader.Url, out var current) || !ReferenceEquals(current, reader))
                return;

            _readers.Remove(reader.Url);

            if (reader.Error == null) return;

            _strikes.TryGetValue(reader.Url, out var strikes);
            strikes++;
            _strikes[reader.Url] = strikes;

            if (strikes > Pauses.Length)
            {
                // Gave up: nothing asks for this camera again until a person forces a retry
                _pausedUntil[reader.Url] = DateTime.MaxValue;
                Console.WriteLine(
                    $"Camera offline: {reader.Url}. Stopped asking after {strikes} attempts. " +
                    "It is tried again only when someone presses Test Connection.");
            }
            else
            {
                var pause = Pauses[strikes - 1];
                _pausedUntil[reader.Url] = DateTime.UtcNow + pause;
                Console.WriteLine(
                    $"Camera offline: {reader.Url} (attempt {strikes} of {Pauses.Length + 1}). " +
                    $"Next try in {pause.TotalSeconds:0} s.");
            }
        }
    }
}