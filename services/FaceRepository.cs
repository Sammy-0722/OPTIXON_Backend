using System.Data.Common;
using Npgsql;
using Npgsql.PostgresTypes;
using Pgvector;
using Pgvector.Npgsql;
using static AIManagement.Api.Repositories.FaceRepository;

namespace AIManagement.Api.Repositories
{
    public record PersonSummary(
        int Id,
        string EmployeeId,
        string Name,
        int EmbeddingCount);

    public class FaceRepository
    {
        private readonly NpgsqlDataSource _dataSource;
        private const int MaxEmbeddingsPerPerson = 20;

        public FaceRepository(IConfiguration configuration)
        {
            var connectionString =
                configuration.GetConnectionString("FaceDatabase");

            var builder = new NpgsqlDataSourceBuilder(connectionString);

            builder.UseVector();

            _dataSource = builder.Build();
        }

        // Save a new embedding for a person identified by employee ID.
        // If the employee already exists, the embedding is added to them.
        public async Task SaveEmbeddingAsync(
            string employeeId,
            string name,
            float[] embedding)
        {
            if (string.IsNullOrWhiteSpace(employeeId))
                throw new ArgumentException("Employee ID is required.");

            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException("Person name is required.");

            if (embedding == null || embedding.Length != 512)
                throw new ArgumentException(
                    "Embedding must contain exactly 512 values.");

            await using var connection =
                await _dataSource.OpenConnectionAsync();

            await using var transaction =
                await connection.BeginTransactionAsync();

            try
            {
                // Find the existing employee or create a new one.
                const string personQuery = """
                    INSERT INTO "People" ("EmployeeId", "Name")
                    VALUES (@employeeId, @name)
                    ON CONFLICT ("EmployeeId")
                    DO UPDATE SET "EmployeeId" = "People"."EmployeeId"
                    RETURNING "Id";
                    """;

                int personId;

                await using (var command =
                    new NpgsqlCommand(personQuery, connection, transaction))
                {
                    command.Parameters.AddWithValue("employeeId", employeeId.Trim());
                    command.Parameters.AddWithValue("name", name.Trim());

                    personId = Convert.ToInt32(
                        await command.ExecuteScalarAsync());
                }

                // Insert the new face embedding.
                const string embeddingQuery = """
                    INSERT INTO "FaceEmbeddings"
                        ("PersonId", "Embedding")
                    VALUES
                        (@personId, @embedding);
                    """;

                await using (var command =
                    new NpgsqlCommand(embeddingQuery, connection, transaction))
                {
                    command.Parameters.AddWithValue("personId", personId);
                    command.Parameters.AddWithValue(
                        "embedding", new Vector(embedding));

                    await command.ExecuteNonQueryAsync();
                }

                // Keep only the latest N embeddings for this person.
                const string cleanupQuery = """
                    DELETE FROM "FaceEmbeddings"
                    WHERE "Id" IN (
                        SELECT "Id"
                        FROM "FaceEmbeddings"
                        WHERE "PersonId" = @personId
                        ORDER BY "CreatedAt" DESC, "Id" DESC
                        OFFSET @maxEmbeddings
                    );
                    """;

                await using (var command =
                    new NpgsqlCommand(cleanupQuery, connection, transaction))
                {
                    command.Parameters.AddWithValue("personId", personId);
                    command.Parameters.AddWithValue(
                        "maxEmbeddings", MaxEmbeddingsPerPerson);

                    await command.ExecuteNonQueryAsync();
                }

                await transaction.CommitAsync();
            }
            catch
            {
                await transaction.RollbackAsync();
                throw;
            }
        }

        // Find the closest matching embedding in the database.
        public async Task<(string Name, string EmployeeId, float Similarity)?> FindMatchAsync(
            float[] embedding)
        {
            if (embedding == null || embedding.Length != 512)
                throw new ArgumentException(
                    "Embedding must contain exactly 512 values.");

            const string query = """
        SELECT
            p."Name",
            p."EmployeeId",
            1 - (fe."Embedding" <=> @embedding) AS similarity
        FROM "FaceEmbeddings" fe
        INNER JOIN "People" p
            ON p."Id" = fe."PersonId"
        ORDER BY fe."Embedding" <=> @embedding
        LIMIT 1;
        """;

            await using var connection =
                await _dataSource.OpenConnectionAsync();

            await using var command =
                new NpgsqlCommand(query, connection);

            command.Parameters.AddWithValue(
                "embedding", new Vector(embedding));

            await using var reader = await command.ExecuteReaderAsync();

            if (await reader.ReadAsync())
            {
                string name = reader.GetString(0);
                string employeeId = reader.GetString(1);
                float similarity = Convert.ToSingle(reader.GetDouble(2));

                return (name, employeeId, similarity);
            }

            return null;
        }
        // GUARD QUERY: best similarity between a new embedding and
        // everything already stored for one employee.
        // Returns null if the employee has no embeddings.
        public async Task<float?> GetBestSimilarityForPersonAsync(
            string employeeId,
            float[] embedding)
        {
            if (string.IsNullOrWhiteSpace(employeeId))
                throw new ArgumentException("Employee ID is required.");

            if (embedding == null || embedding.Length != 512)
                throw new ArgumentException(
                    "Embedding must contain exactly 512 values.");

            const string query = """
                SELECT MAX(1 - (fe."Embedding" <=> @embedding))
                FROM "FaceEmbeddings" fe
                INNER JOIN "People" p
                    ON p."Id" = fe."PersonId"
                WHERE p."EmployeeId" = @employeeId;
                """;

            await using var connection =
                await _dataSource.OpenConnectionAsync();

            await using var command =
                new NpgsqlCommand(query, connection);

            command.Parameters.AddWithValue("employeeId", employeeId.Trim());
            command.Parameters.AddWithValue(
                "embedding", new Vector(embedding));

            var result = await command.ExecuteScalarAsync();

            if (result == null || result is DBNull)
                return null;

            return Convert.ToSingle(result);
        }

        // Get all registered people (names only, kept for existing code).
        public async Task<List<string>> GetPeopleAsync()
        {
            const string query = """
                SELECT "Name"
                FROM "People"
                ORDER BY "Name";
                """;

            var people = new List<string>();

            await using var connection =
                await _dataSource.OpenConnectionAsync();

            await using var command =
                new NpgsqlCommand(query, connection);

            await using var reader = await command.ExecuteReaderAsync();

            while (await reader.ReadAsync())
            {
                people.Add(reader.GetString(0));
            }

            return people;
        }

        // Roster list: id, employee ID, name and how many embeddings each has.
        public async Task<List<PersonSummary>> GetPeopleDetailsAsync()
        {
            const string query = """
                SELECT
                    p."Id",
                    p."EmployeeId",
                    p."Name",
                    COUNT(fe."Id")
                FROM "People" p
                LEFT JOIN "FaceEmbeddings" fe
                    ON fe."PersonId" = p."Id"
                GROUP BY p."Id", p."EmployeeId", p."Name"
                ORDER BY p."Name";
                """;

            var people = new List<PersonSummary>();

            await using var connection =
                await _dataSource.OpenConnectionAsync();

            await using var command =
                new NpgsqlCommand(query, connection);

            await using var reader = await command.ExecuteReaderAsync();

            while (await reader.ReadAsync())
            {
                people.Add(new PersonSummary(
                    reader.GetInt32(0),
                    reader.GetString(1),
                    reader.GetString(2),
                    Convert.ToInt32(reader.GetInt64(3))));
            }

            return people;
        }

        // Delete one employee and all their embeddings.
        // Returns false if no such employee exists.
        public async Task<bool> DeletePersonAsync(string employeeId)
        {
            if (string.IsNullOrWhiteSpace(employeeId))
                throw new ArgumentException("Employee ID is required.");

            await using var connection =
                await _dataSource.OpenConnectionAsync();

            await using var transaction =
                await connection.BeginTransactionAsync();

            try
            {
                const string deleteEmbeddings = """
                    DELETE FROM "FaceEmbeddings"
                    WHERE "PersonId" IN (
                        SELECT "Id" FROM "People"
                        WHERE "EmployeeId" = @employeeId
                    );
                    """;

                await using (var command =
                    new NpgsqlCommand(deleteEmbeddings, connection, transaction))
                {
                    command.Parameters.AddWithValue("employeeId", employeeId.Trim());
                    await command.ExecuteNonQueryAsync();
                }

                const string deletePerson = """
                    DELETE FROM "People"
                    WHERE "EmployeeId" = @employeeId;
                    """;

                int deleted;

                await using (var command =
                    new NpgsqlCommand(deletePerson, connection, transaction))
                {
                    command.Parameters.AddWithValue("employeeId", employeeId.Trim());
                    deleted = await command.ExecuteNonQueryAsync();
                }

                await transaction.CommitAsync();
                return deleted > 0;
            }
            catch
            {
                await transaction.RollbackAsync();
                throw;
            }
        }
        public record SightingSummary(
    int Id, string EmployeeId, string Name,
    string Source, float Similarity, DateTime SeenAt);

        public async Task<int> AddSightingAsync(
    string employeeId, string name, string source,
    float similarity, string filePath)
        {
            const string query = """
        INSERT INTO "Sightings"
            ("EmployeeId", "Name", "Source", "Similarity", "FilePath", "SeenAt")
        VALUES
            (@employeeId, @name, @source, @similarity, @filePath, @seenAt)
        RETURNING "Id";
        """;

            await using var connection = await _dataSource.OpenConnectionAsync();
            await using var command = new NpgsqlCommand(query, connection);

            command.Parameters.AddWithValue("employeeId", employeeId);
            command.Parameters.AddWithValue("name", name);
            command.Parameters.AddWithValue("source", source);
            command.Parameters.AddWithValue("similarity", similarity);
            command.Parameters.AddWithValue("filePath", filePath);
            command.Parameters.AddWithValue("seenAt", DateTime.UtcNow);

            return Convert.ToInt32(await command.ExecuteScalarAsync());
        }

        public async Task<bool> UpdateSightingAsync(int id, float similarity, string filePath)
        {
            const string query = """
        UPDATE "Sightings"
        SET "Similarity" = @similarity, "FilePath" = @filePath, "SeenAt" = @seenAt
        WHERE "Id" = @id;
        """;

            await using var connection = await _dataSource.OpenConnectionAsync();
            await using var command = new NpgsqlCommand(query, connection);

            command.Parameters.AddWithValue("id", id);
            command.Parameters.AddWithValue("similarity", similarity);
            command.Parameters.AddWithValue("filePath", filePath);
            command.Parameters.AddWithValue("seenAt", DateTime.UtcNow);

            return await command.ExecuteNonQueryAsync() > 0;
        }

        // Deletes old rows and returns their file paths so the files can be removed too.
        public async Task<List<string>> DeleteSightingsOlderThanAsync(DateTime cutoffUtc)
        {
            const string query = """
        DELETE FROM "Sightings"
        WHERE "SeenAt" < @cutoff
        RETURNING "FilePath";
        """;

            var paths = new List<string>();

            await using var connection = await _dataSource.OpenConnectionAsync();
            await using var command = new NpgsqlCommand(query, connection);
            command.Parameters.AddWithValue("cutoff", cutoffUtc);

            await using var reader = await command.ExecuteReaderAsync();
            while (await reader.ReadAsync())
                paths.Add(reader.GetString(0));

            return paths;
        }

        public async Task<List<SightingSummary>> GetSightingsAsync(int limit)
        {
            const string query = """
        SELECT "Id", "EmployeeId", "Name", "Source", "Similarity", "SeenAt"
        FROM "Sightings"
        ORDER BY "SeenAt" DESC
        LIMIT @limit;
        """;

            var list = new List<SightingSummary>();

            await using var connection = await _dataSource.OpenConnectionAsync();
            await using var command = new NpgsqlCommand(query, connection);
            command.Parameters.AddWithValue("limit", limit);

            await using var reader = await command.ExecuteReaderAsync();
            while (await reader.ReadAsync())
            {
                list.Add(new SightingSummary(
                    reader.GetInt32(0), reader.GetString(1), reader.GetString(2),
                    reader.GetString(3), reader.GetFloat(4), reader.GetDateTime(5)));
            }

            return list;
        }

        public async Task<string?> GetSightingPathAsync(int id)
        {
            const string query = """SELECT "FilePath" FROM "Sightings" WHERE "Id" = @id;""";

            await using var connection = await _dataSource.OpenConnectionAsync();
            await using var command = new NpgsqlCommand(query, connection);
            command.Parameters.AddWithValue("id", id);

            return await command.ExecuteScalarAsync() as string;
        }

        public record HourlyCount(DateTime Hour, int Recognized, int Unknown);

        public record DashboardSummary(
            int RegisteredPeople,
            int NewPeopleThisWeek,
            int Recognized24h,
            int Unknown24h,
            int UniquePeople24h,
            double AverageSimilarity,
            List<HourlyCount> Hourly,
            List<SightingSummary> Recent);

        public async Task<DashboardSummary> GetDashboardSummaryAsync()
        {
            int registered = 0, newThisWeek = 0, recognized = 0, unknown = 0, unique = 0;
            double averageSimilarity = 0;
            var hourly = new List<HourlyCount>();

            await using var connection = await _dataSource.OpenConnectionAsync();

            const string totalsQuery = """
                SELECT
                    (SELECT COUNT(*) FROM "People"),
                    (SELECT COUNT(*) FROM (
                        SELECT "PersonId" FROM "FaceEmbeddings"
                        GROUP BY "PersonId"
                        HAVING MIN("CreatedAt") >= now() - interval '7 days') AS recent_people),
                    COUNT(*) FILTER (WHERE "EmployeeId" <> 'UNKNOWN'),
                    COUNT(*) FILTER (WHERE "EmployeeId" = 'UNKNOWN'),
                    COUNT(DISTINCT "EmployeeId") FILTER (WHERE "EmployeeId" <> 'UNKNOWN'),
                    COALESCE(AVG("Similarity") FILTER (WHERE "EmployeeId" <> 'UNKNOWN'), 0)
                FROM "Sightings"
                WHERE "SeenAt" >= now() - interval '24 hours';
                """;

            await using (var command = new NpgsqlCommand(totalsQuery, connection))
            await using (var reader = await command.ExecuteReaderAsync())
            {
                if (await reader.ReadAsync())
                {
                    registered = Convert.ToInt32(reader.GetInt64(0));
                    newThisWeek = Convert.ToInt32(reader.GetInt64(1));
                    recognized = Convert.ToInt32(reader.GetInt64(2));
                    unknown = Convert.ToInt32(reader.GetInt64(3));
                    unique = Convert.ToInt32(reader.GetInt64(4));
                    averageSimilarity = reader.GetDouble(5);
                }
            }

            const string hourlyQuery = """
                SELECT g.hour,
                       COUNT(s."Id") FILTER (WHERE s."EmployeeId" <> 'UNKNOWN'),
                       COUNT(s."Id") FILTER (WHERE s."EmployeeId" = 'UNKNOWN')
                FROM generate_series(
                        date_trunc('hour', now() - interval '23 hours'),
                        date_trunc('hour', now()),
                        interval '1 hour') AS g(hour)
                LEFT JOIN "Sightings" s
                       ON s."SeenAt" >= g.hour AND s."SeenAt" < g.hour + interval '1 hour'
                GROUP BY g.hour
                ORDER BY g.hour;
                """;

            await using (var command = new NpgsqlCommand(hourlyQuery, connection))
            await using (var reader = await command.ExecuteReaderAsync())
            {
                while (await reader.ReadAsync())
                {
                    hourly.Add(new HourlyCount(
                        reader.GetDateTime(0),
                        Convert.ToInt32(reader.GetInt64(1)),
                        Convert.ToInt32(reader.GetInt64(2))));
                }
            }

            var recent = await GetSightingsAsync(8);

            return new DashboardSummary(
                registered, newThisWeek, recognized, unknown, unique,
                averageSimilarity, hourly, recent);
        }
    }
}

















































