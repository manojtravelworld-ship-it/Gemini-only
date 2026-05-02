import axios from 'axios';

/**
 * LocalDB implementation that syncs with a PostgreSQL backend via Express API.
 */
export class LocalDB {
  private static instance: LocalDB;
  private data: Record<string, any[]> = {
    clients: []
  };

  private constructor() {}

  public static getInstance(): LocalDB {
    if (!LocalDB.instance) {
      LocalDB.instance = new LocalDB();
    }
    return LocalDB.instance;
  }

  public async init() {
    try {
      const response = await axios.get('/api/clients');
      this.data.clients = response.data;
      return true;
    } catch (err) {
      console.error("Failed to initialize database from PG:", err);
      return false;
    }
  }

  public async query(sql: string): Promise<any[]> {
    // For simplicity, we just return the cached data if it's a "SELECT * FROM clients"
    // In a real app, this would execute the SQL on the server.
    if (sql.toLowerCase().includes('from clients')) {
      const response = await axios.get('/api/clients');
      this.data.clients = response.data;
      return this.data.clients;
    }
    return [];
  }

  public async run(sql: string, params: any[]) {
    if (sql.toLowerCase().includes('insert into clients')) {
      const [name, phone, case_number, court, next_date, purpose] = params;
      try {
        await axios.post('/api/clients', {
          name, phone, case_number, court, next_date, purpose
        });
        await this.init(); // Refresh local cache
      } catch (err) {
        console.error("Failed to persist to PG:", err);
      }
    }
  }
}
