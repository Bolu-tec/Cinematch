import { useState, useEffect, useRef, useCallback } from "react";

const API_BASE = "http://localhost:5000/api";
const IMG_BASE = "https://image.tmdb.org/t/p/w92";

// ── UI Helpers ────────────────────────────────────────────────────────────────
const GENRE_COLORS = {
  28:"#e85d04",10751:"#2dc653",16:"#7209b7",99:"#3a86ff",
  35:"#06d6a0",80:"#ef233c",18:"#8338ec",14:"#fb5607",
  36:"#8d99ae",27:"#d00000",10402:"#f72585",9648:"#480ca8",
  10749:"#ff006e",878:"#3a0ca3",53:"#c9184a",10752:"#6c757d",
  37:"#a52a2a",12:"#f77f00",10770:"#f4a261"
};

function GenreTag({ name, genreId }) {
  const color = GENRE_COLORS[genreId] || "#666";
  return (
    <span style={{
      background: color+"22", color, border:`1px solid ${color}55`,
      borderRadius:4, padding:"1px 7px", fontSize:11, fontWeight:600,
      letterSpacing:"0.03em", whiteSpace:"nowrap"
    }}>{name}</span>
  );
}

function StarRating({ value, onChange }) {
  const [hover, setHover] = useState(0);
  return (
    <div style={{ display:"flex", gap:2 }}>
      {[1,2,3,4,5].map(s => (
        <span key={s} onClick={() => onChange(s)}
          onMouseEnter={() => setHover(s)} onMouseLeave={() => setHover(0)}
          style={{ cursor:"pointer", fontSize:18, color: s<=(hover||value) ? "#f5a623":"#333", transition:"color 0.1s" }}>★</span>
      ))}
    </div>
  );
}

function ScoreBar({ score }) {
  const pct = Math.round(score * 100);
  const color = pct>70 ? "#06d6a0" : pct>45 ? "#f5a623" : "#8338ec";
  return (
    <div style={{ display:"flex", alignItems:"center", gap:8 }}>
      <div style={{ flex:1, height:4, background:"#1a1a2e", borderRadius:2, overflow:"hidden" }}>
        <div style={{ width:`${pct}%`, height:"100%", background:color, borderRadius:2, transition:"width 0.6s ease" }}/>
      </div>
      <span style={{ fontSize:11, color, fontWeight:700, minWidth:32 }}>{pct}%</span>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [query, setQuery]             = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [searching, setSearching]     = useState(false);
  const [watched, setWatched]         = useState([]);
  const [ratings, setRatings]         = useState({});
  const [recs, setRecs]               = useState([]);
  const [mode, setMode]               = useState("");
  const [trained, setTrained]         = useState(false);
  const [training, setTraining]       = useState(false);
  const [activeTab, setActiveTab]     = useState("watched");
  const [genreMap, setGenreMap]       = useState({});
  const [error, setError]             = useState("");
  const inputRef    = useRef();
  const debounceRef = useRef();

  useEffect(() => {
    fetch(`${API_BASE}/genres`)
      .then(r => r.json())
      .then(setGenreMap)
      .catch(() => setError("Could not connect to backend. Make sure Flask is running on port 5000."));
  }, []);

  const searchMovies = useCallback((q) => {
    clearTimeout(debounceRef.current);
    if (q.trim().length < 2) { setSuggestions([]); return; }
    debounceRef.current = setTimeout(async () => {
      setSearching(true);
      try {
        const res  = await fetch(`${API_BASE}/search?q=${encodeURIComponent(q)}`);
        const data = await res.json();
        const watchedIds = new Set(watched.map(m => m.id));
        setSuggestions(data.filter(m => !watchedIds.has(m.id)));
        setError("");
      } catch {
        setError("Could not connect to backend. Make sure Flask is running on port 5000.");
      } finally {
        setSearching(false);
      }
    }, 350);
  }, [watched]);

  useEffect(() => { searchMovies(query); }, [query, searchMovies]);

  function addMovie(movie) {
    setWatched(w => [...w, movie]);
    setRatings(r => ({ ...r, [movie.id]: 3 }));
    setQuery(""); setSuggestions([]);
    setTrained(false);
    inputRef.current?.focus();
  }

  function removeMovie(id) {
    setWatched(w => w.filter(m => m.id !== id));
    setRatings(r => { const n={...r}; delete n[id]; return n; });
    setTrained(false);
  }

  async function train() {
    setTraining(true); setError("");
    try {
      const res = await fetch(`${API_BASE}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ watched, ratings }),
      });
      const data = await res.json();
      if (data.error) { setError(data.error); return; }
      setRecs(data.recommendations);
      setMode(data.mode);
      setTrained(true);
      setActiveTab("recs");
    } catch {
      setError("Training failed. Make sure Flask backend is running.");
    } finally {
      setTraining(false);
    }
  }

  return (
    <div style={{
      minHeight:"100vh", background:"#0a0a14", fontFamily:"'Georgia',serif", color:"#e8e8f0",
      backgroundImage:"radial-gradient(ellipse at 20% 20%,#1a0533 0%,transparent 50%),radial-gradient(ellipse at 80% 80%,#0d1f3c 0%,transparent 50%)"
    }}>
      <div style={{ borderBottom:"1px solid #ffffff0f", padding:"28px 0 22px", textAlign:"center" }}>
        <div style={{ fontSize:11, letterSpacing:"0.3em", color:"#8338ec", textTransform:"uppercase", marginBottom:8 }}>
          scikit-learn · Flask · TMDB
        </div>
        <h1 style={{ margin:0, fontSize:38, fontWeight:400, letterSpacing:"-0.02em", color:"#f0eeff" }}>🎬 CineMatch</h1>
        <p style={{ margin:"8px 0 0", color:"#666680", fontSize:14 }}>Python-powered movie recommendations — TF-IDF + kNN via scikit-learn</p>
      </div>

      <div style={{ maxWidth:900, margin:"0 auto", padding:"32px 20px" }}>
        {error && (
          <div style={{ background:"#ef233c22", border:"1px solid #ef233c55", borderRadius:8, padding:"10px 16px", marginBottom:20, color:"#ef233c", fontSize:13 }}>
            ⚠️ {error}
          </div>
        )}

        {/* Search */}
        <div style={{ marginBottom:32 }}>
          <label style={{ display:"block", fontSize:12, letterSpacing:"0.15em", color:"#8338ec", textTransform:"uppercase", marginBottom:10 }}>
            Search Any Movie Ever Made
          </label>
          <div style={{ position:"relative" }}>
            <input ref={inputRef} value={query} onChange={e => setQuery(e.target.value)}
              placeholder="e.g. The Lion King, Inception, Your Name…"
              style={{ width:"100%", boxSizing:"border-box", background:"#12122a", border:"1px solid #2a2a4a", borderRadius:10, padding:"14px 18px", fontSize:15, color:"#e8e8f0", outline:"none" }}
            />
            {searching && <div style={{ position:"absolute", right:16, top:"50%", transform:"translateY(-50%)", color:"#8338ec", fontSize:12 }}>searching…</div>}
            {suggestions.length > 0 && (
              <div style={{ position:"absolute", top:"calc(100% + 6px)", left:0, right:0, zIndex:100, background:"#16162e", border:"1px solid #2a2a4a", borderRadius:10, overflow:"hidden", boxShadow:"0 20px 60px #00000088" }}>
                {suggestions.map(m => (
                  <div key={m.id} onClick={() => addMovie(m)}
                    style={{ padding:"10px 14px", cursor:"pointer", borderBottom:"1px solid #1e1e38", display:"flex", gap:12, alignItems:"center" }}
                    onMouseEnter={e => e.currentTarget.style.background="#1e1e40"}
                    onMouseLeave={e => e.currentTarget.style.background="transparent"}>
                    {m.poster_path
                      ? <img src={IMG_BASE+m.poster_path} alt="" style={{ width:36, height:54, objectFit:"cover", borderRadius:4, flexShrink:0 }}/>
                      : <div style={{ width:36, height:54, background:"#1e1e38", borderRadius:4, flexShrink:0, display:"flex", alignItems:"center", justifyContent:"center" }}>🎬</div>
                    }
                    <div>
                      <div style={{ fontWeight:600, fontSize:14 }}>{m.title} <span style={{ color:"#555570", fontWeight:400 }}>({m.release_date?.slice(0,4)||"?"})</span></div>
                      <div style={{ display:"flex", gap:4, marginTop:4, flexWrap:"wrap" }}>
                        {(m.genre_ids||[]).slice(0,3).map(id => <GenreTag key={id} genreId={id} name={genreMap[id]||"…"}/>)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Tabs */}
        {watched.length > 0 && (
          <>
            <div style={{ display:"flex", gap:4, marginBottom:24, borderBottom:"1px solid #1e1e38" }}>
              {["watched","recs"].map(tab => (
                <button key={tab} onClick={() => setActiveTab(tab)} style={{
                  background:"none", border:"none", cursor:"pointer", padding:"8px 20px", fontSize:13, fontFamily:"inherit",
                  color: activeTab===tab ? "#f0eeff":"#555570",
                  borderBottom: activeTab===tab ? "2px solid #8338ec":"2px solid transparent",
                  letterSpacing:"0.05em", textTransform:"uppercase", marginBottom:-1,
                }}>
                  {tab==="watched" ? `Watched (${watched.length})` : `Recommendations${trained?` (${recs.length})`:""}`}
                </button>
              ))}
            </div>

            {/* Watched */}
            {activeTab==="watched" && (
              <div>
                <div style={{ display:"grid", gap:12, marginBottom:28 }}>
                  {watched.map(m => (
                    <div key={m.id} style={{ background:"#12122a", border:"1px solid #1e1e38", borderRadius:12, padding:"12px 16px", display:"flex", alignItems:"center", gap:14 }}>
                      {m.poster_path
                        ? <img src={IMG_BASE+m.poster_path} alt="" style={{ width:40, height:60, objectFit:"cover", borderRadius:6, flexShrink:0 }}/>
                        : <div style={{ width:40, height:60, background:"#1e1e38", borderRadius:6, flexShrink:0, display:"flex", alignItems:"center", justifyContent:"center" }}>🎬</div>
                      }
                      <div style={{ flex:1, minWidth:0 }}>
                        <div style={{ fontWeight:600, fontSize:15, marginBottom:4 }}>{m.title} <span style={{ color:"#555570", fontWeight:400, fontSize:13 }}>({m.release_date?.slice(0,4)||"?"})</span></div>
                        <div style={{ display:"flex", gap:4, flexWrap:"wrap" }}>
                          {(m.genre_ids||[]).slice(0,4).map(id => <GenreTag key={id} genreId={id} name={genreMap[id]||"…"}/>)}
                        </div>
                      </div>
                      <div style={{ display:"flex", flexDirection:"column", alignItems:"flex-end", gap:4 }}>
                        <StarRating value={ratings[m.id]||3} onChange={v => { setRatings(r=>({...r,[m.id]:v})); setTrained(false); }}/>
                        <span style={{ fontSize:11, color:"#444460" }}>Your rating</span>
                      </div>
                      <button onClick={() => removeMovie(m.id)}
                        style={{ background:"none", border:"none", color:"#444460", cursor:"pointer", fontSize:18, padding:"0 4px" }}
                        onMouseEnter={e => e.currentTarget.style.color="#ef233c"}
                        onMouseLeave={e => e.currentTarget.style.color="#444460"}>✕</button>
                    </div>
                  ))}
                </div>
                <button onClick={train} disabled={training} style={{
                  width:"100%", padding:"16px", borderRadius:12, border:"none",
                  background: training ? "#1e1e38":"linear-gradient(135deg,#8338ec,#3a86ff)",
                  color:"#fff", fontSize:15, fontFamily:"inherit", fontWeight:600,
                  cursor: training ? "not-allowed":"pointer", letterSpacing:"0.05em",
                }}>
                  {training ? "⚙️  Running scikit-learn model…" : trained ? "🔄  Retrain Model" : "🚀  Train Model & Get Recommendations"}
                </button>
              </div>
            )}

            {/* Recommendations */}
            {activeTab==="recs" && (
              <div>
                {!trained ? (
                  <div style={{ textAlign:"center", padding:"60px 20px", color:"#444460" }}>
                    <div style={{ fontSize:48, marginBottom:16 }}>🎯</div>
                    <div style={{ fontSize:15 }}>Rate your movies then hit <strong style={{ color:"#8338ec" }}>Train Model</strong>.</div>
                  </div>
                ) : (
                  <>
                    <div style={{ marginBottom:20, padding:"12px 16px", background:"#8338ec11", border:"1px solid #8338ec33", borderRadius:8, fontSize:13, color:"#a070f0" }}>
                      {mode === "hybrid"
                        ? <>✨ <strong>Hybrid mode</strong> — scikit-learn TF-IDF vectorizer + k-Nearest Neighbours (k={Math.min(5, watched.length)}). Rare genres weighted higher, ratings train the kNN model.</>
                        : <>⚡ <strong>Cosine similarity mode</strong> — add {5 - watched.length} more movie{5-watched.length===1?"":"s"} to unlock full kNN hybrid.</>
                      }
                    </div>
                    <div style={{ display:"grid", gap:12 }}>
                      {recs.map((m,i) => (
                        <div key={m.id} style={{ background:"#12122a", border:"1px solid #1e1e38", borderRadius:12, padding:"14px 16px", display:"flex", gap:14, alignItems:"flex-start", borderLeft: i<3?"3px solid #8338ec":"3px solid #1e1e38" }}>
                          {m.poster_path
                            ? <img src={`https://image.tmdb.org/t/p/w92${m.poster_path}`} alt="" style={{ width:50, height:75, objectFit:"cover", borderRadius:6, flexShrink:0 }}/>
                            : <div style={{ width:50, height:75, background:"#1e1e38", borderRadius:6, flexShrink:0, display:"flex", alignItems:"center", justifyContent:"center" }}>🎬</div>
                          }
                          <div style={{ flex:1, minWidth:0 }}>
                            <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:4, flexWrap:"wrap" }}>
                              <span style={{ fontSize:11, color:"#8338ec", fontWeight:700 }}>#{i+1}</span>
                              <span style={{ fontWeight:600, fontSize:15 }}>{m.title}</span>
                              <span style={{ color:"#555570", fontSize:13 }}>({m.release_date?.slice(0,4)||"?"})</span>
                            </div>
                            <div style={{ display:"flex", gap:4, flexWrap:"wrap", marginBottom:6 }}>
                              {(m.genre_ids||[]).slice(0,4).map(id => <GenreTag key={id} genreId={id} name={genreMap[id]||"…"}/>)}
                            </div>
                            <div style={{ fontSize:13, color:"#8888aa", lineHeight:1.5, marginBottom:8 }}>
                              {m.overview?.slice(0,130)}{(m.overview?.length||0)>130?"…":""}
                            </div>
                            <ScoreBar score={m.score}/>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}
          </>
        )}

        {watched.length===0 && (
          <div style={{ textAlign:"center", padding:"80px 20px", color:"#333350" }}>
            <div style={{ fontSize:56, marginBottom:16 }}>🍿</div>
            <div style={{ fontSize:16, marginBottom:8 }}>Search and add any movie you've watched</div>
            <div style={{ fontSize:13 }}>Rate them, then the Python ML model finds your next favourites</div>
          </div>
        )}
      </div>
    </div>
  );
}
