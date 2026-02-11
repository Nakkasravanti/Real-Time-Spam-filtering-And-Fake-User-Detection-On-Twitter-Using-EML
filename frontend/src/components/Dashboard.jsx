import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const API_BASE = '/api';

export default function Dashboard() {
    const [logs, setLogs] = useState([]);
    const [activeModule, setActiveModule] = useState('home'); // 'home', 'text', 'search', 'verify', 'scan'

    // Module States
    const [textInput, setTextInput] = useState('');
    const [loading, setLoading] = useState(false);

    // Scan/Analyze State
    const [datasetPath, setDatasetPath] = useState('tweets');
    const [stats, setStats] = useState(null);
    const [chartData, setChartData] = useState([]);

    // Manual Check State
    const [manualFollowers, setManualFollowers] = useState('');
    const [manualFollowing, setManualFollowing] = useState('');
    const [manualResult, setManualResult] = useState(null);

    // Search State
    const [searchUsername, setSearchUsername] = useState('');

    const addLog = (msg) => {
        setLogs(prev => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev]);
    };

    const checkText = async () => {
        if (!textInput) return;
        setLoading(true);
        try {
            const res = await axios.post(`${API_BASE}/predict/text`, { text: textInput });
            addLog(`Text Analysis Result: ${res.data.result}`);
        } catch (err) {
            addLog(`Error: ${err.message}`);
        }
        setLoading(false);
    };

    const searchUser = async () => {
        if (!searchUsername) return;
        setLoading(true);
        try {
            const res = await axios.post(`${API_BASE}/search/user`, { username: searchUsername });
            const data = res.data;

            if (data.error && (data.error.includes("not found") || data.error.includes("No data"))) {
                addLog(`'${searchUsername}' not in local DB. Redirecting to Twitter...`);
                window.open(`https://twitter.com/${searchUsername}`, '_blank');
            }
            else if (data.error) {
                alert(data.error);
                addLog(`Search Error: ${data.error}`);
            } else {
                if (data.is_fake) {
                    alert(`WARNING: Account '${searchUsername}' is marked as FAKE!`);
                    addLog(`Search: ${searchUsername} is FAKE.`);
                } else {
                    addLog(`Search: ${searchUsername} is Real. Redirecting...`);
                    window.open(`https://twitter.com/${searchUsername}`, '_blank');
                }
            }
        } catch (err) {
            addLog(`Error: ${err.message}`);
        }
        setLoading(false);
    };

    const handleUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);
        setLoading(true);
        addLog(`Uploading ${file.name}...`);
        try {
            const res = await axios.post(`${API_BASE}/upload`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            if (res.data.error) {
                addLog(`Upload Error: ${res.data.error}`);
            } else {
                addLog(`Upload Success: ${file.name} added.`);
                setDatasetPath(res.data.path);
            }
        } catch (err) {
            addLog(`Upload Error: ${err.message}`);
        }
        setLoading(false);
    };

    const analyzeFolder = async () => {
        setLoading(true);
        addLog(`Starting Folder Analysis on: ${datasetPath}...`);
        try {
            const res = await axios.post(`${API_BASE}/analyze`, { path: datasetPath });
            const d = res.data;
            if (d.error) {
                addLog(`Error: ${d.error}`);
            } else {
                addLog(`Analysis Complete. Total: ${d.total_accounts}, Fake: ${d.fake_accounts}, Spam: ${d.spam_accounts}`);
                setStats(d);
                d.details.forEach(detail => {
                    addLog(`User: ${detail.username} | Fake: ${detail.is_fake} | Spam: ${detail.is_spam}`);
                });
            }
        } catch (err) {
            addLog(`Error: ${err.message}`);
        }
        setLoading(false);
    };

    const trainModel = async (algo) => {
        setLoading(true);
        addLog(`Training/Running ${algo.toUpperCase()}...`);
        try {
            const res = await axios.post(`${API_BASE}/train/${algo}`);
            const d = res.data;
            if (d.error) {
                addLog(`Error: ${d.error}`);
            } else {
                addLog(`${d.algorithm} Results: Accuracy: ${d.accuracy.toFixed(2)}% | Precision: ${d.precision.toFixed(2)}`);
                setChartData(prev => {
                    const exists = prev.find(item => item.name === d.algorithm);
                    if (exists) return prev.map(item => item.name === d.algorithm ? { name: d.algorithm, accuracy: d.accuracy } : item);
                    return [...prev, { name: d.algorithm, accuracy: d.accuracy }];
                });
            }
        } catch (err) {
            addLog(`Error: ${err.message}`);
        }
        setLoading(false);
    };

    const verifyUser = async () => {
        if (!manualFollowers || !manualFollowing) return;
        setLoading(true);
        try {
            const res = await axios.post(`${API_BASE}/check/user`, {
                followers: manualFollowers,
                following: manualFollowing
            });
            setManualResult(res.data);
            addLog(`Manual Check: ${res.data.message}`);
        } catch (err) {
            addLog(`Error: ${err.message}`);
        }
        setLoading(false);
    };

    const LogPanel = () => (
        <div className="glass-panel" style={{ marginTop: '20px', maxHeight: '200px', display: 'flex', flexDirection: 'column' }}>
            <h4 style={{ margin: '0 0 10px 0', fontSize: '0.9em', color: '#a0aec0' }}>Activity Log</h4>
            <div style={{
                flex: 1,
                overflowY: 'auto',
                background: 'rgba(0,0,0,0.3)',
                borderRadius: '8px',
                padding: '12px',
                fontFamily: 'monospace',
                fontSize: '0.85em',
            }}>
                {logs.map((log, i) => (
                    <div key={i} style={{ marginBottom: '4px', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '4px' }}>
                        {log}
                    </div>
                ))}
                {logs.length === 0 && <div style={{ color: '#64748b' }}>Ready...</div>}
            </div>
        </div>
    );

    return (
        <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
            {/* Marquee Section */}
            <div className="marquee-container">
                <div className="marquee-content">
                    <span>Aditya College of Engineering and Technology &nbsp;&nbsp;&bull;&nbsp;&nbsp; Aditya College of Engineering and Technology &nbsp;&nbsp;&bull;&nbsp;&nbsp; Aditya College of Engineering and Technology &nbsp;&nbsp;&bull;&nbsp;&nbsp; Aditya College of Engineering and Technology</span>
                </div>
            </div>

            <div className="layout-container" style={{ flex: 1, paddingTop: '20px' }}>
                <header style={{ marginBottom: '40px', textAlign: 'center' }}>
                    <h1 style={{
                        fontSize: '3rem',
                        color: '#fff',
                        textShadow: '0 0 10px #00e5ff, 0 0 20px #00e5ff', /* Neon Cyan Glow */
                        marginBottom: '10px'
                    }}>
                        Spam Shield AI
                    </h1>
                    <p style={{ color: '#e2e8f0', fontSize: '1.2rem', fontWeight: '500', textShadow: '0 1px 2px rgba(0,0,0,0.8)' }}>
                        Advanced Social Network Spam & Fake User Detection
                    </p>
                </header>

                <div style={{ minHeight: '400px' }}>

                    {/* HOME SCREEN */}
                    {activeModule === 'home' && (
                        <div className="home-menu-grid module-container">
                            <div className="menu-btn" onClick={() => setActiveModule('scan')}>
                                <h3>Directory Scan & Stats</h3>
                                <p>Analyze datasets, view statistics, and train ML models.</p>
                            </div>
                            <div className="menu-btn" onClick={() => setActiveModule('search')}>
                                <h3>Search Account</h3>
                                <p>Find and verify user accounts from the database.</p>
                            </div>
                            <div className="menu-btn" onClick={() => setActiveModule('verify')}>
                                <h3>Verify User (Manual)</h3>
                                <p>Check account credibility based on followers/following ratio.</p>
                            </div>
                            <div className="menu-btn" onClick={() => setActiveModule('text')}>
                                <h3>Live Text Check</h3>
                                <p>Analyze tweets or messages for spam content in real-time.</p>
                            </div>
                        </div>
                    )}

                    {/* TEXT CHECK MODULE */}
                    {activeModule === 'text' && (
                        <div className="module-container" style={{ maxWidth: '600px', margin: '0 auto' }}>
                            <div onClick={() => setActiveModule('home')} className="back-btn">← Back to Home</div>
                            <div className="glass-panel">
                                <h3>Live Text Check</h3>
                                <textarea
                                    rows="6"
                                    placeholder="Enter tweet content here..."
                                    value={textInput}
                                    onChange={(e) => setTextInput(e.target.value)}
                                    style={{ marginBottom: '12px' }}
                                />
                                <button className="btn-primary" style={{ width: '100%' }} onClick={checkText} disabled={loading}>
                                    {loading ? 'Analyzing...' : 'Check for Spam'}
                                </button>
                            </div>
                            <LogPanel />
                        </div>
                    )}

                    {/* SEARCH MODULE */}
                    {activeModule === 'search' && (
                        <div className="module-container" style={{ maxWidth: '600px', margin: '0 auto' }}>
                            <div onClick={() => setActiveModule('home')} className="back-btn">← Back to Home</div>
                            <div className="glass-panel">
                                <h3>Search & Redirect</h3>
                                <div style={{ marginBottom: '15px' }}>
                                    <label style={{ display: 'block', marginBottom: '8px', color: '#a0aec0' }}>Username</label>
                                    <input
                                        type="text"
                                        placeholder="Enter Username"
                                        value={searchUsername}
                                        onChange={(e) => setSearchUsername(e.target.value)}
                                    />
                                </div>
                                <button className="btn-secondary" style={{ width: '100%' }} onClick={searchUser} disabled={loading}>
                                    {loading ? 'Searching...' : 'Search Database'}
                                </button>
                            </div>
                            <LogPanel />
                        </div>
                    )}

                    {/* VERIFY MODULE */}
                    {activeModule === 'verify' && (
                        <div className="module-container" style={{ maxWidth: '600px', margin: '0 auto' }}>
                            <div onClick={() => setActiveModule('home')} className="back-btn">← Back to Home</div>
                            <div className="glass-panel">
                                <h3>Verify User (Manual)</h3>
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '12px' }}>
                                    <div>
                                        <label style={{ display: 'block', marginBottom: '8px', color: '#a0aec0' }}>Followers</label>
                                        <input
                                            type="number"
                                            value={manualFollowers}
                                            onChange={(e) => setManualFollowers(e.target.value)}
                                        />
                                    </div>
                                    <div>
                                        <label style={{ display: 'block', marginBottom: '8px', color: '#a0aec0' }}>Following</label>
                                        <input
                                            type="number"
                                            value={manualFollowing}
                                            onChange={(e) => setManualFollowing(e.target.value)}
                                        />
                                    </div>
                                </div>
                                <button className="btn-primary" style={{ width: '100%' }} onClick={verifyUser} disabled={loading}>
                                    Verify Account
                                </button>
                                {manualResult && (
                                    <div style={{
                                        marginTop: '12px',
                                        padding: '12px',
                                        borderRadius: '8px',
                                        background: manualResult.is_fake ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)',
                                        border: `1px solid ${manualResult.is_fake ? '#ef4444' : '#10b981'}`,
                                        textAlign: 'center',
                                        fontWeight: 'bold',
                                        color: manualResult.is_fake ? '#fecaca' : '#a7f3d0'
                                    }}>
                                        {manualResult.message}
                                    </div>
                                )}
                            </div>
                            <LogPanel />
                        </div>
                    )}

                    {/* SCAN & STATS MODULE */}
                    {activeModule === 'scan' && (
                        <div className="module-container">
                            <div onClick={() => setActiveModule('home')} className="back-btn">← Back to Home</div>
                            <div className="grid-cols-2">
                                <div>
                                    <div className="glass-panel" style={{ marginBottom: '24px' }}>
                                        <h3>Dataset Controls</h3>
                                        <div style={{ marginBottom: '12px' }}>
                                            <input
                                                type="text"
                                                placeholder="Path to dataset"
                                                value={datasetPath}
                                                onChange={(e) => setDatasetPath(e.target.value)}
                                                style={{ marginBottom: '10px' }}
                                            />
                                            <div style={{ display: 'flex', gap: '10px' }}>
                                                <input type="file" accept=".json" onChange={handleUpload} disabled={loading} style={{ flex: 1 }} />
                                            </div>
                                        </div>
                                        <button className="btn-primary" onClick={analyzeFolder} disabled={loading} style={{ width: '100%' }}>
                                            Run Full Analysis
                                        </button>
                                    </div>

                                    <div className="glass-panel">
                                        <h3>Train Models</h3>
                                        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                                            <button className="btn-secondary" onClick={() => trainModel('naive_bayes')}>NB Algo</button>
                                            <button className="btn-secondary" onClick={() => trainModel('svm')}>SVM Algo</button>
                                            <button className="btn-primary" onClick={() => trainModel('elm')}>ELM (Best)</button>
                                        </div>
                                    </div>
                                </div>

                                <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                                    {stats && (
                                        <div className="glass-panel" style={{ background: 'rgba(16, 185, 129, 0.1)' }}>
                                            <h3>Results</h3>
                                            <div style={{ display: 'flex', justifyContent: 'space-around', textAlign: 'center' }}>
                                                <div>
                                                    <div style={{ fontSize: '2em' }}>{stats.total_accounts}</div>
                                                    <div style={{ fontSize: '0.8em', opacity: 0.7 }}>Total</div>
                                                </div>
                                                <div>
                                                    <div style={{ fontSize: '2em', color: '#ef4444' }}>{stats.fake_accounts}</div>
                                                    <div style={{ fontSize: '0.8em', opacity: 0.7 }}>Fake</div>
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    <div className="glass-panel" style={{ flex: 1, minHeight: '300px' }}>
                                        <ResponsiveContainer width="100%" height="100%">
                                            <BarChart data={chartData}>
                                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                                                <XAxis dataKey="name" stroke="#a0aec0" />
                                                <YAxis stroke="#a0aec0" />
                                                <Tooltip contentStyle={{ backgroundColor: '#1e293b' }} />
                                                <Bar dataKey="accuracy" fill="#8884d8" radius={[4, 4, 0, 0]} />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>
                            </div>
                            <LogPanel />
                        </div>
                    )}

                </div>
            </div>

            {/* Footer Section */}
            <footer className="app-footer">
                <div className="dev-credits">
                    <div style={{ marginBottom: '10px' }}>Developed by</div>
                    <div className="dev-names">
                        <span className="dev-name">Nakka Sravanti (22MH1A0541)</span>
                        <span className="dev-name">Yerubandi Renuka Lakshmi (22MH1A0555)</span>
                        <span className="dev-name">Dara Durga Jyothi Prasad (22MH1A0512)</span>
                        <span className="dev-name">Vijjapu Siva Satya Durga Vara Prasad (22MH1A0575)</span>
                    </div>
                </div>
            </footer>
        </div>
    );
}
