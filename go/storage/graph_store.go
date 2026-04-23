package storage

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	_ "github.com/mattn/go-sqlite3"
)

const graphSchema = `
CREATE TABLE IF NOT EXISTS graph_memory_nodes (
    id             TEXT PRIMARY KEY,
    summary        TEXT,
    tier           TEXT,
    salience       REAL,
    valence        REAL,
    compaction_gen INTEGER,
    created_at     TEXT
);

CREATE TABLE IF NOT EXISTS graph_entity_nodes (
    id   TEXT PRIMARY KEY,
    name TEXT,
    type TEXT
);

CREATE TABLE IF NOT EXISTS graph_relates_to (
    from_id           TEXT NOT NULL,
    to_id             TEXT NOT NULL,
    weight            REAL,
    relationship_type TEXT,
    created_at        TEXT,
    FOREIGN KEY (from_id) REFERENCES graph_memory_nodes(id),
    FOREIGN KEY (to_id)   REFERENCES graph_memory_nodes(id)
);

CREATE INDEX IF NOT EXISTS idx_relates_from ON graph_relates_to(from_id);
CREATE INDEX IF NOT EXISTS idx_relates_to   ON graph_relates_to(to_id);

CREATE TABLE IF NOT EXISTS graph_mentions (
    memory_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    weight    REAL,
    FOREIGN KEY (memory_id) REFERENCES graph_memory_nodes(id),
    FOREIGN KEY (entity_id) REFERENCES graph_entity_nodes(id)
);

CREATE INDEX IF NOT EXISTS idx_mentions_mem ON graph_mentions(memory_id);

CREATE TABLE IF NOT EXISTS graph_evolved_from (
    new_id        TEXT NOT NULL,
    source_id     TEXT NOT NULL,
    compaction_id TEXT,
    created_at    TEXT,
    FOREIGN KEY (new_id)    REFERENCES graph_memory_nodes(id),
    FOREIGN KEY (source_id) REFERENCES graph_memory_nodes(id)
);

CREATE INDEX IF NOT EXISTS idx_evolved_new ON graph_evolved_from(new_id);
CREATE INDEX IF NOT EXISTS idx_evolved_src ON graph_evolved_from(source_id);
`

// GraphStore implements a graph database for memory relationships using SQLite
// with recursive CTEs instead of Kuzu.
type GraphStore struct {
	graphDir string
	db       *sql.DB
}

// NewGraphStore creates a new GraphStore.
func NewGraphStore(graphDir string) *GraphStore {
	return &GraphStore{graphDir: graphDir}
}

// Initialize opens the graph database and creates the schema.
func (g *GraphStore) Initialize(ctx context.Context) error {
	if err := os.MkdirAll(filepath.Dir(g.graphDir), 0o755); err != nil {
		return fmt.Errorf("creating graph dir parent: %w", err)
	}
	if err := os.MkdirAll(g.graphDir, 0o755); err != nil {
		return fmt.Errorf("creating graph dir: %w", err)
	}

	dbPath := filepath.Join(g.graphDir, "graph.db")
	db, err := sql.Open("sqlite3", dbPath+"?_journal_mode=WAL&_busy_timeout=5000")
	if err != nil {
		return fmt.Errorf("opening graph database: %w", err)
	}
	g.db = db

	if _, err := g.db.ExecContext(ctx, graphSchema); err != nil {
		return fmt.Errorf("creating graph schema: %w", err)
	}
	return nil
}

// Close closes the graph database.
func (g *GraphStore) Close() error {
	if g.db != nil {
		return g.db.Close()
	}
	return nil
}

// DB returns the underlying database. Panics if not initialized.
func (g *GraphStore) DB() *sql.DB {
	if g.db == nil {
		panic("GraphStore not initialized — call Initialize() first")
	}
	return g.db
}

// ---------- Node operations ----------

// AddMemoryNode inserts or updates a memory node.
func (g *GraphStore) AddMemoryNode(ctx context.Context, memoryID, summary, tier string, salience, valence float64, compactionGen int, createdAt string) error {
	_, err := g.DB().ExecContext(ctx,
		`INSERT INTO graph_memory_nodes (id, summary, tier, salience, valence, compaction_gen, created_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?)
		 ON CONFLICT(id) DO UPDATE SET summary=excluded.summary, tier=excluded.tier,
		   salience=excluded.salience, valence=excluded.valence,
		   compaction_gen=excluded.compaction_gen, created_at=excluded.created_at`,
		memoryID, summary, tier, salience, valence, compactionGen, createdAt,
	)
	return err
}

// AddEntityNode inserts or updates an entity node.
func (g *GraphStore) AddEntityNode(ctx context.Context, entityID, name, entityType string) error {
	_, err := g.DB().ExecContext(ctx,
		`INSERT INTO graph_entity_nodes (id, name, type) VALUES (?, ?, ?)
		 ON CONFLICT(id) DO UPDATE SET name=excluded.name, type=excluded.type`,
		entityID, name, entityType,
	)
	return err
}

// ---------- Edge operations ----------

// AddRelatesTo creates a RELATES_TO edge between two memory nodes.
func (g *GraphStore) AddRelatesTo(ctx context.Context, fromID, toID string, weight float64, relationshipType, createdAt string) error {
	_, err := g.DB().ExecContext(ctx,
		`INSERT INTO graph_relates_to (from_id, to_id, weight, relationship_type, created_at)
		 VALUES (?, ?, ?, ?, ?)`,
		fromID, toID, weight, relationshipType, createdAt,
	)
	return err
}

// AddMentions creates a MENTIONS edge from a memory to an entity.
func (g *GraphStore) AddMentions(ctx context.Context, memoryID, entityID string, weight float64) error {
	_, err := g.DB().ExecContext(ctx,
		`INSERT INTO graph_mentions (memory_id, entity_id, weight) VALUES (?, ?, ?)`,
		memoryID, entityID, weight,
	)
	return err
}

// AddEvolvedFrom creates an EVOLVED_FROM edge (new memory evolved from source).
func (g *GraphStore) AddEvolvedFrom(ctx context.Context, newID, sourceID, compactionID, createdAt string) error {
	_, err := g.DB().ExecContext(ctx,
		`INSERT INTO graph_evolved_from (new_id, source_id, compaction_id, created_at)
		 VALUES (?, ?, ?, ?)`,
		newID, sourceID, compactionID, createdAt,
	)
	return err
}

// ---------- Queries ----------

// RelatedMemory is a result from graph traversal.
type RelatedMemory struct {
	ID       string
	Summary  string
	Tier     string
	Salience float64
	Depth    int
}

// GetRelatedMemories traverses RELATES_TO edges up to maxDepth hops using recursive CTE.
func (g *GraphStore) GetRelatedMemories(ctx context.Context, memoryID string, maxDepth int, minWeight float64) ([]RelatedMemory, error) {
	query := fmt.Sprintf(`
		WITH RECURSIVE traversal(id, depth) AS (
			SELECT to_id, 1 FROM graph_relates_to WHERE from_id = ? AND weight >= ?
			UNION
			SELECT r.to_id, t.depth + 1
			FROM graph_relates_to r
			JOIN traversal t ON r.from_id = t.id
			WHERE t.depth < %d AND r.weight >= ?
		)
		SELECT DISTINCT m.id, m.summary, m.tier, m.salience, t.depth
		FROM traversal t
		JOIN graph_memory_nodes m ON t.id = m.id
		WHERE m.id != ?
		ORDER BY t.depth ASC
	`, maxDepth)

	rows, err := g.DB().QueryContext(ctx, query, memoryID, minWeight, minWeight, memoryID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []RelatedMemory
	for rows.Next() {
		var r RelatedMemory
		var summary sql.NullString
		if err := rows.Scan(&r.ID, &summary, &r.Tier, &r.Salience, &r.Depth); err != nil {
			return nil, err
		}
		if summary.Valid {
			r.Summary = summary.String
		}
		results = append(results, r)
	}
	return results, nil
}

// MemoryEntity is an entity mentioned by a memory.
type MemoryEntity struct {
	ID     string
	Name   string
	Type   string
	Weight float64
}

// GetMemoryEntities returns all entities mentioned by a memory.
func (g *GraphStore) GetMemoryEntities(ctx context.Context, memoryID string) ([]MemoryEntity, error) {
	rows, err := g.DB().QueryContext(ctx,
		`SELECT e.id, e.name, e.type, m.weight
		 FROM graph_mentions m
		 JOIN graph_entity_nodes e ON m.entity_id = e.id
		 WHERE m.memory_id = ?`,
		memoryID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []MemoryEntity
	for rows.Next() {
		var e MemoryEntity
		if err := rows.Scan(&e.ID, &e.Name, &e.Type, &e.Weight); err != nil {
			return nil, err
		}
		results = append(results, e)
	}
	return results, nil
}

// LineageNode is a node in an evolution lineage.
type LineageNode struct {
	ID      string
	Summary string
	Gen     int
	Depth   int
}

// GetEvolutionLineage traces the full lineage of a compacted memory back to originals using recursive CTE.
func (g *GraphStore) GetEvolutionLineage(ctx context.Context, memoryID string) ([]LineageNode, error) {
	rows, err := g.DB().QueryContext(ctx,
		`WITH RECURSIVE lineage(id, depth) AS (
			SELECT source_id, 1 FROM graph_evolved_from WHERE new_id = ?
			UNION
			SELECT e.source_id, l.depth + 1
			FROM graph_evolved_from e
			JOIN lineage l ON e.new_id = l.id
			WHERE l.depth < 10
		)
		SELECT m.id, m.summary, m.compaction_gen, l.depth
		FROM lineage l
		JOIN graph_memory_nodes m ON l.id = m.id
		ORDER BY l.depth`,
		memoryID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []LineageNode
	for rows.Next() {
		var n LineageNode
		var summary sql.NullString
		if err := rows.Scan(&n.ID, &summary, &n.Gen, &n.Depth); err != nil {
			return nil, err
		}
		if summary.Valid {
			n.Summary = summary.String
		}
		results = append(results, n)
	}
	return results, nil
}

// GetEdgeCount counts RELATES_TO edges connected to a memory (both directions).
func (g *GraphStore) GetEdgeCount(ctx context.Context, memoryID string) (int, error) {
	var count int
	err := g.DB().QueryRowContext(ctx,
		`SELECT COUNT(*) FROM graph_relates_to WHERE from_id = ? OR to_id = ?`,
		memoryID, memoryID,
	).Scan(&count)
	return count, err
}

// ReplicateEdgesToNewNode copies all RELATES_TO edges from source memories to a new compacted node.
// Skips edges between source nodes (they are being merged).
func (g *GraphStore) ReplicateEdgesToNewNode(ctx context.Context, sourceIDs []string, newID string) error {
	srcSet := make(map[string]bool)
	for _, id := range sourceIDs {
		srcSet[id] = true
	}

	for _, srcID := range sourceIDs {
		// Outgoing edges
		rows, err := g.DB().QueryContext(ctx,
			`SELECT to_id, weight, relationship_type, created_at
			 FROM graph_relates_to WHERE from_id = ?`, srcID,
		)
		if err != nil {
			return err
		}
		for rows.Next() {
			var toID, rtype, cat string
			var weight float64
			if err := rows.Scan(&toID, &weight, &rtype, &cat); err != nil {
				rows.Close()
				return err
			}
			if !srcSet[toID] && toID != newID {
				// Ignore errors from duplicate edges
				_ = g.AddRelatesTo(ctx, newID, toID, weight, rtype, cat)
			}
		}
		rows.Close()

		// Incoming edges
		rows, err = g.DB().QueryContext(ctx,
			`SELECT from_id, weight, relationship_type, created_at
			 FROM graph_relates_to WHERE to_id = ?`, srcID,
		)
		if err != nil {
			return err
		}
		for rows.Next() {
			var fromID, rtype, cat string
			var weight float64
			if err := rows.Scan(&fromID, &weight, &rtype, &cat); err != nil {
				rows.Close()
				return err
			}
			if !srcSet[fromID] && fromID != newID {
				_ = g.AddRelatesTo(ctx, fromID, newID, weight, rtype, cat)
			}
		}
		rows.Close()
	}

	return nil
}

// PathExists checks if a path exists between two memory nodes via RELATES_TO edges using recursive CTE.
func (g *GraphStore) PathExists(ctx context.Context, fromID, toID string, maxHops int) (bool, error) {
	query := fmt.Sprintf(`
		WITH RECURSIVE path(id, depth) AS (
			SELECT to_id, 1 FROM graph_relates_to WHERE from_id = ?
			UNION
			SELECT r.to_id, p.depth + 1
			FROM graph_relates_to r
			JOIN path p ON r.from_id = p.id
			WHERE p.depth < %d
		)
		SELECT COUNT(*) FROM path WHERE id = ? LIMIT 1
	`, maxHops)

	var count int
	err := g.DB().QueryRowContext(ctx, query, fromID, toID).Scan(&count)
	if err != nil {
		return false, err
	}
	return count > 0, nil
}

// UpdateMemoryTier updates the tier of a memory node in the graph.
func (g *GraphStore) UpdateMemoryTier(ctx context.Context, memoryID, tier string) error {
	_, err := g.DB().ExecContext(ctx,
		"UPDATE graph_memory_nodes SET tier = ? WHERE id = ?", tier, memoryID,
	)
	return err
}

// GetAllMemoryNodeIDs returns all memory node IDs, optionally filtered by tiers.
func (g *GraphStore) GetAllMemoryNodeIDs(ctx context.Context, tiers []string) ([]string, error) {
	query := "SELECT id FROM graph_memory_nodes"
	var args []interface{}
	if len(tiers) > 0 {
		ph := make([]string, len(tiers))
		for i, t := range tiers {
			ph[i] = "?"
			args = append(args, t)
		}
		query += fmt.Sprintf(" WHERE tier IN (%s)", strings.Join(ph, ","))
	}

	rows, err := g.DB().QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, nil
}
