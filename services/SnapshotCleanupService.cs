using AIManagement.Api.Repositories;
using AIManagement.Api.services;

namespace AIManagement.Api.Services;

public sealed class SnapshotCleanupService : BackgroundService
{
    private static readonly TimeSpan Retention = TimeSpan.FromHours(24);
    private static readonly TimeSpan Interval = TimeSpan.FromMinutes(10);

    private readonly IServiceScopeFactory _scopes;
    private readonly SnapshotService _snapshots;

    public SnapshotCleanupService(IServiceScopeFactory scopes, SnapshotService snapshots)
    {
        _scopes = scopes;
        _snapshots = snapshots;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        using var timer = new PeriodicTimer(Interval);

        do
        {
            try { await CleanAsync(); }
            catch (Exception ex) { Console.WriteLine($"Snapshot cleanup failed: {ex.Message}"); }
        }
        while (await timer.WaitForNextTickAsync(stoppingToken));
    }

    private async Task CleanAsync()
    {
        var cutoff = DateTime.UtcNow - Retention;

        // 1. Delete old rows and the files they point to
        using (var scope = _scopes.CreateScope())
        {
            var repository = scope.ServiceProvider.GetRequiredService<FaceRepository>();
            var paths = await repository.DeleteSightingsOlderThanAsync(cutoff);

            foreach (var path in paths)
            {
                var full = _snapshots.Resolve(path);
                if (full != null && File.Exists(full)) File.Delete(full);
            }
        }

        // 2. Safety sweep: any leftover file older than the retention period
        foreach (var file in Directory.EnumerateFiles(_snapshots.Root, "*.jpg", SearchOption.AllDirectories))
        {
            if (File.GetLastWriteTimeUtc(file) < cutoff) File.Delete(file);
        }

        // 3. Remove empty date folders
        foreach (var dir in Directory.GetDirectories(_snapshots.Root))
        {
            if (!Directory.EnumerateFileSystemEntries(dir).Any()) Directory.Delete(dir);
        }
    }
}