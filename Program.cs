using AIManagement.Api.Repositories;
using AIManagement.Api.services;
using AIManagement.Api.Services;

// Quiet the camera library's own log lines. These must run before any camera is opened.
Environment.SetEnvironmentVariable("OPENCV_FFMPEG_LOGLEVEL", "-8");
Environment.SetEnvironmentVariable("OPENCV_LOG_LEVEL", "SILENT");

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();
builder.Services.AddOpenApi();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// Face recognition
builder.Services.AddSingleton<FaceInferenceService>();
builder.Services.AddSingleton<FaceRepository>();
builder.Services.AddSingleton<RtspFaceRecognitionService>();
builder.Services.AddSingleton<LocalCameraTestService>();
builder.Services.AddSingleton<StreamHub>();
builder.Services.AddSingleton<SnapshotService>();
builder.Services.AddHostedService<SnapshotCleanupService>();
builder.Services.Configure<StickyIdentityOptions>(
    builder.Configuration.GetSection("StickyIdentity"));
builder.Services.AddSingleton<StickyIdentityService>();

// Phone detection
builder.Services.Configure<PhoneDetectionOptions>(
    builder.Configuration.GetSection("PhoneDetection"));
builder.Services.AddSingleton<PhoneDetectionService>();
builder.Services.AddSingleton<PhoneUseService>();
builder.Services.AddSingleton<PhoneSnapshotService>();

// Cameras
builder.Services.AddSingleton<CameraStore>();
builder.Services.AddSingleton<CameraMonitor>();
builder.Services.AddHostedService(sp => sp.GetRequiredService<CameraMonitor>());

builder.Services.AddCors(options =>
{
    options.AddPolicy("frontend", policy =>
    {
        policy.WithOrigins("http://localhost:3000")
            .AllowAnyHeader()
            .AllowAnyMethod();
    });
});

var app = builder.Build();

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
    app.MapOpenApi();
}

app.UseDefaultFiles();
app.UseStaticFiles();

app.UseHttpsRedirection();
app.UseCors("frontend");

app.UseAuthorization();

app.MapControllers();

app.Run();