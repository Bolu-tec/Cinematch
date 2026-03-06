import { useState, useEffect, useRef, useCallback } from "react";

const TMDB_KEY = "527b096928be9f627a2f7ce5be8eae31";
const TMDB_BASE = "https://api.themoviedb.org/3";
const IMG_BASE = "https://image.tmdb.org/t/p/w92";

// ── ML Engine: TF-IDF + kNN ───────────────────────────────────────────────────

// TF-IDF: genres that appear in fewer movies get higher weight (more informative)
function computeTFIDF(allMovies, allGenreIds) {
  const N = allMovies.length;
  // DF: how many movies contain each genre
  const df = {};
  allGenreIds.forEach(id => {
    df[id] = allMovies.filter(m => (m.genre_ids || []).includes(id)).length;
  });
  // IDF: log(N / df) — rare genres score higher
  const idf = {};
  allGenreIds.forEach(id => {
    idf[id] = df[id] > 0 ? Math.log(N / df[id]) : 0;
  });
  // Build TF-IDF vector for each movie
  const vectors = {};
  allMovies.forEach(m => {
    const genreCount = (m.genre_ids || []).length || 1;
    vectors[m.id] = allGenreIds.map(id => {
      const tf = (m.genre_ids || []).includes(id) ? 1 / genreCount : 0; // TF
      return tf * idf[id]; // TF-IDF
    });
  });
  return vectors;
}

function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, v, i) => sum + v * b[i], 0);
  const magA = Math.sqrt(a.reduce((s, v) => s + v * v, 0));
  const magB = Math.sqrt(b.reduce((s, v) => s + v * v, 0));
  return magA && magB ? dot / (magA * magB) : 0;
}

// kNN: find the K most similar watched movies to a candidate, aggregate their ratings
function knnScore(candidateVec, watchedVectors, watchedMovies, ratings, k = 5) {
  const neighbours = watchedMovies
    .map(m => ({
      similarity: cosineSimilarity(candidateVec, watchedVectors[m.id] || []),
      rating: ratings[m.id] || 3,
    }))
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, k);

  const totalSim = neighbours.reduce((s, n) => s + n.similarity, 0);
  if (totalSim === 0) return 0;
  // Weighted average rating from K nearest neighbours
  const weightedRating = neighbours.reduce((s, n) => s + n.similarity * n.rating, 0) / totalSim;
  // Normalise to 0-1
  return weightedRating / 5;
}

const KNN_THRESHOLD = 5; // switch to hybrid once user has watched this many movies

function getRecommendations(watchedMovies, ratings, candidates) {
  if (!watchedMovies.length || !candidates.length) return [];

  const useHybrid   = watchedMovies.length >= KNN_THRESHOLD;
  const allMovies   = [...watchedMovies, ...candidates];
  const allGenreIds = [...new Set(allMovies.flatMap(m => m.genre_ids || []))];

  const tfidfVectors = computeTFIDF(allMovies, allGenreIds);

  const profileLen = allGenreIds.length;
  const profile    = new Array(profileLen).fill(0);
  let totalWeight  = 0;
  watchedMovies.forEach(m => {
    const weight = (ratings[m.id] || 3) / 5;
    (tfidfVectors[m.id] || []).forEach((v, i) => { profile[i] += v * weight; });
    totalWeight += weight;
  });
  if (totalWeight > 0) profile.forEach((_, i) => { profile[i] /= totalWeight; });

  const watchedIds = new Set(watchedMovies.map(m => m.id));
  const K = Math.min(5, watchedMovies.length);

  return candidates
    .filter(m => !watchedIds.has(m.id))
    .map(m => {
      const vec        = tfidfVectors[m.id] || [];
      const profileSim = cosineSimilarity(profile, vec);
      let score;
      if (useHybrid) {
        const knn = knnScore(vec, tfidfVectors, watchedMovies, ratings, K);
        score = 0.5 * profileSim + 0.5 * knn;
      } else {
        score = profileSim;
      }
      return { ...m, score, useHybrid };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, 10);
}

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

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const [query, setQuery]             = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [searching, setSearching]     = useState(false);
  const [watched, setWatched]         = useState([]);
  const [ratings, setRatings]         = useState({});
  const [recs, setRecs]               = useState([]);
  const [trained, setTrained]         = useState(false);
  const [training, setTraining]       = useState(false);
  const [activeTab, setActiveTab]     = useState("watched");
  const [genreMap, setGenreMap]       = useState({});
  const [error, setError]             = useState("");
  const inputRef   = useRef();
  const debounceRef = useRef();

  useEffect(() => {
    fetch(`${TMDB_BASE}/genre/movie/list?api_key=${TMDB_KEY}`)
      .then(r => r.json())
      .then(data => {
        const map = {};
        (data.genres || []).forEach(g => { map[g.id] = g.name; });
        setGenreMap(map);
      }).catch(() => {});
  }, []);

  const searchMovies = useCallback((q) => {
    clearTimeout(debounceRef.current);
    if (q.trim().length < 2) { setSuggestions([]); return; }
    debounceRef.current = setTimeout(async () => {
      setSearching(true);
      try {
        const res  = await fetch(`${TMDB_BASE}/search/movie?api_key=${TMDB_KEY}&query=${encodeURIComponent(q)}&include_adult=false`);
        const data = await res.json();
        const watchedIds = new Set(watched.map(m => m.id));
        setSuggestions((data.results || []).filter(m => !watchedIds.has(m.id)).slice(0, 7));
        setError("");
      } catch {
        setError("Couldn't reach TMDB. Check your connection.");
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
      const fetches = [
        ...[1,2,3,4,5].map(p => fetch(`${TMDB_BASE}/movie/popular?api_key=${TMDB_KEY}&page=${p}`).then(r=>r.json())),
        ...[1,2,3].map(p => fetch(`${TMDB_BASE}/movie/top_rated?api_key=${TMDB_KEY}&page=${p}`).then(r=>r.json())),
        ...[1,2].map(p => fetch(`${TMDB_BASE}/movie/now_playing?api_key=${TMDB_KEY}&page=${p}`).then(r=>r.json())),
      ];
      const pages  = await Promise.all(fetches);
      const all    = pages.flatMap(p => p.results || []);
      const seen   = new Set();
      const unique = all.filter(m => { if(seen.has(m.id)) return false; seen.add(m.id); return true; });
      setRecs(getRecommendations(watched, ratings, unique));
      setTrained(true);
      setActiveTab("recs");
    } catch {
      setError("Training failed — please try again.");
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
        <div style={{ fontSize:11, letterSpacing:"0.3em", color:"#8338ec", textTransform:"uppercase", marginBottom:8 }}>Content-Based ML · Powered by TMDB</div>
        <h1 style={{ margin:0, fontSize:38, fontWeight:400, letterSpacing:"-0.02em", color:"#f0eeff" }}>🎬 CineMatch</h1>
        <p style={{ margin:"8px 0 0", color:"#666680", fontSize:14 }}>Search any movie, rate what you've watched, train your model</p>
      </div>

      <div style={{ maxWidth:900, margin:"0 auto", padding:"32px 20px" }}>
        {error && (
          <div style={{ background:"#ef233c22", border:"1px solid #ef233c55", borderRadius:8, padding:"10px 16px", marginBottom:20, color:"#ef233c", fontSize:13 }}>
            ⚠️ {error}
          </div>
        )}

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
                  {training ? "⚙️  Fetching movies & training model…" : trained ? "🔄  Retrain Model" : "🚀  Train Model & Get Recommendations"}
                </button>
              </div>
            )}

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
                      {recs[0]?.useHybrid
  ? <>✨ <strong>Hybrid mode</strong> — TF-IDF + k-Nearest Neighbours (k={Math.min(5,watched.length)}). Rare genres are weighted higher and your ratings directly train the model.</>
  : <>⚡ <strong>Cosine similarity mode</strong> — add {5 - watched.length} more movie{5-watched.length===1?"":"s"} to unlock the full TF-IDF + kNN hybrid model.</>
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
            <div style={{ fontSize:13 }}>Rate them, then train the model to discover your next favourites</div>
          </div>
        )}
      </div>
    </div>
  );
}
