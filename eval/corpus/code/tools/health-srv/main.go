// health-srv: tiny HTTP server that reports whether the app is alive.
// GET /healthz returns 200 with a JSON body, 503 if the db check fails.
package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"time"
)

type status struct {
	Service   string `json:"service"`
	Version   string `json:"version"`
	Uptime    string `json:"uptime"`
	DBOK      bool   `json:"db_ok"`
	CheckedAt string `json:"checked_at"`
}

var startedAt = time.Now()

// pingDB is a stand-in for the real check: it GETs the db proxy's status URL.
func pingDB(ctx context.Context, addr string) bool {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, addr, nil)
	if err != nil {
		return false
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode < 500
}

func healthz(w http.ResponseWriter, r *http.Request) {
	dbAddr := os.Getenv("HEALTH_DB_URL")
	dbOK := true
	if dbAddr != "" {
		dbOK = pingDB(r.Context(), dbAddr)
	}

	body := status{
		Service:   "billing-api",
		Version:   os.Getenv("APP_VERSION"),
		Uptime:    time.Since(startedAt).Round(time.Second).String(),
		DBOK:      dbOK,
		CheckedAt: time.Now().UTC().Format(time.RFC3339),
	}

	w.Header().Set("Content-Type", "application/json")
	if !dbOK {
		w.WriteHeader(http.StatusServiceUnavailable)
	}
	// an encode error here isn't worth handling, the client just gets a short body
	_ = json.NewEncoder(w).Encode(body)
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8081"
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", healthz)

	srv := &http.Server{
		Addr:              ":" + port,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	log.Printf("health server listening on %s", srv.Addr)
	if err := srv.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}
