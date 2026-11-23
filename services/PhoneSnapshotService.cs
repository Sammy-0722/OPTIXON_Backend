using System.Collections.Concurrent;

namespace AIManagement.Api.Services;

public sealed record SavedSnapshot(string FileName, string Url, DateTime CapturedAt);

public sealed class PhoneSnapshotService
{
    // Must match the folder and URL prefix your face module already uses
    private const string FolderName = "Snapshots";
    private const string UrlPrefix = "/snapshots";

    private static readonly TimeSpan Cooldown = TimeSpan.FromSeconds(5);

    private readonly string _folder;
    private readonly ConcurrentDictionary<string, DateTime> _lastSaved = new();

    public PhoneSnapshotService(IWebHostEnvironment environment)
    {
        _folder = Path.Combine(environment.ContentRootPath, FolderName);
        Directory.CreateDirectory(_folder);
    }

    // Saves a JPEG. When a cooldownKey is given (e.g. the stream URL), saves at most
    // once per cooldown and returns null if it was skipped.
    public async Task<SavedSnapshot?> SaveAsync(
        byte[] jpeg,
        string? cooldownKey,
        string baseUrl,
        CancellationToken cancellationToken = default)
    {
        var now = DateTime.UtcNow;

        if (cooldownKey != null)
        {
            if (_lastSaved.TryGetValue(cooldownKey, out var last) && now - last < Cooldown)
                return null;

            _lastSaved[cooldownKey] = now;
        }

        var fileName = $"phone_{now:yyyyMMdd_HHmmss_fff}.jpg";
        await File.WriteAllBytesAsync(Path.Combine(_folder, fileName), jpeg, cancellationToken);

        return new SavedSnapshot(fileName, $"{baseUrl}{UrlPrefix}/{fileName}", now);
    }
}