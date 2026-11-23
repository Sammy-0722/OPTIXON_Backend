using System.Text.Json;

namespace AIManagement.Api.services;

public sealed class CameraRecord
{
    public string Id { get; set; } = "";
    public string TenantId { get; set; } = "default";
    public string Name { get; set; } = "";
    public string Location { get; set; } = "";
    public string Url { get; set; } = "";          // full link, login included (temporary)
    public List<string> Modules { get; set; } = new();
}

public sealed class CameraStore
{
    private readonly string _path;
    private readonly object _lock = new();
    private List<CameraRecord> _cameras = new();

    public CameraStore(IWebHostEnvironment env)
    {
        _path = Path.Combine(env.ContentRootPath, "App_Data", "cameras.json");
        Directory.CreateDirectory(Path.GetDirectoryName(_path)!);
        if (File.Exists(_path))
        {
            try { _cameras = JsonSerializer.Deserialize<List<CameraRecord>>(File.ReadAllText(_path)) ?? new(); }
            catch { /* broken file: start empty */ }
        }
    }

    public List<CameraRecord> All()
    {
        lock (_lock) return _cameras.ToList();
    }

    public void ReplaceForTenant(string tenantId, List<CameraRecord> incoming)
    {
        lock (_lock)
        {
            _cameras = _cameras.Where(c => c.TenantId != tenantId).Concat(incoming).ToList();
            File.WriteAllText(_path, JsonSerializer.Serialize(_cameras));
        }
    }
}