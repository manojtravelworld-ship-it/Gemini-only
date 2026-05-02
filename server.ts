import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import pg from "pg";
import dotenv from "dotenv";

dotenv.config();

const { Pool } = pg;

// Lazy initialization of PostgreSQL client
let pool: pg.Pool | null = null;
const fallbackClients: any[] = [];

function getPool() {
  const connectionString = process.env.DATABASE_URL;
  if (!connectionString || connectionString.includes('username:password')) {
    if (connectionString) {
       console.warn("PostgreSQL: Placeholder DATABASE_URL detected. Using in-memory fallback.");
    } else {
       console.warn("PostgreSQL: DATABASE_URL not defined. Using in-memory fallback.");
    }
    return null;
  }

  if (!pool) {
    try {
      pool = new Pool({
        connectionString,
        ssl: connectionString.includes('localhost') || connectionString.includes('127.0.0.1') ? false : { rejectUnauthorized: false },
        connectionTimeoutMillis: 5000, // Fast fail
      });
      // Test connection immediately
      pool.on('error', (err) => {
        console.error('PostgreSQL: Unexpected error on idle client', err);
        pool = null; // Reset pool on critical error
      });
    } catch (err) {
      console.error("PostgreSQL: Pool creation failed:", err);
      return null;
    }
  }
  return pool;
}

async function startServer() {
  const app = express();
  app.use(express.json());
  const PORT = 3000;

  // Initialize Database Schema
  const db = getPool();
  if (db) {
    db.query(`
      CREATE TABLE IF NOT EXISTS clients (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        phone TEXT,
        case_number TEXT,
        court TEXT,
        next_date TEXT,
        purpose TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `).then(() => {
      console.log("PostgreSQL: Schema verification successful.");
    }).catch(err => {
      console.error("PostgreSQL: Schema check failed (Fallback active):", err.message);
    });
  }

  // API Routes
  app.get("/api/clients", async (req, res) => {
    try {
      const db = getPool();
      if (db) {
        const result = await db.query("SELECT * FROM clients ORDER BY created_at DESC");
        return res.json(result.rows);
      }
      throw new Error("No database connection");
    } catch (err) {
      console.warn("PostgreSQL Fetch Failed: Serving from in-memory fallback.");
      res.json(fallbackClients.sort((a, b) => b.id - a.id));
    }
  });

  app.post("/api/clients", async (req, res) => {
    try {
      const { name, phone, case_number, court, next_date, purpose } = req.body;
      const db = getPool();
      if (db) {
        const result = await db.query(
          "INSERT INTO clients (name, phone, case_number, court, next_date, purpose) VALUES ($1, $2, $3, $4, $5, $6) RETURNING *",
          [name, phone, case_number, court, next_date, purpose]
        );
        return res.json(result.rows[0]);
      }
      throw new Error("No database connection");
    } catch (err) {
      console.warn("PostgreSQL Insert Failed: Saving to in-memory fallback.");
      const newClient = {
        id: Date.now(),
        name: req.body.name,
        phone: req.body.phone,
        case_number: req.body.case_number,
        court: req.body.court,
        next_date: req.body.next_date,
        purpose: req.body.purpose,
        created_at: new Date().toISOString()
      };
      fallbackClients.push(newClient);
      res.json(newClient);
    }
  });

  // Vite integration
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
