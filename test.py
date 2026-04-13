body {
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background: linear-gradient(180deg, #f5f7fb 0%, #eef2f7 100%);
    color: #132238;
}

.page-wrap {
    max-width: 1520px;
    margin: 0 auto;
    padding: 24px;
}

.hero {
    background: linear-gradient(135deg, #102a43 0%, #163e63 55%, #1d5f99 100%);
    color: white;
    border-radius: 22px;
    padding: 24px 28px;
    box-shadow: 0 14px 34px rgba(16, 42, 67, 0.18);
    margin-bottom: 20px;
}

.hero h1 {
    margin: 0;
    font-size: 30px;
    letter-spacing: 0.3px;
}

.hero p {
    margin: 8px 0 0 0;
    opacity: 0.9;
    max-width: 1000px;
}

.filter-bar {
    display: grid;
    grid-template-columns: repeat(4, minmax(180px, 1fr));
    gap: 14px;
    margin-bottom: 18px;
}

.filter-box {
    background: white;
    padding: 14px 16px;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(16, 42, 67, 0.08);
}

.filter-label {
    font-size: 12px;
    font-weight: 700;
    color: #52616b;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.cards-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(220px, 1fr));
    gap: 16px;
    margin-bottom: 22px;
}

.kpi-card {
    background: white;
    border-radius: 18px;
    padding: 18px 18px 16px 18px;
    box-shadow: 0 10px 26px rgba(16, 42, 67, 0.08);
    min-height: 128px;
}

.kpi-card-title {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #637381;
    margin-bottom: 10px;
    font-weight: 700;
}

.kpi-card-value {
    font-size: 32px;
    font-weight: 800;
    line-height: 1.05;
    color: #102a43;
    margin-bottom: 10px;
}

.kpi-card-subtitle {
    font-size: 13px;
    color: #52616b;
    line-height: 1.35;
}

.panel {
    background: white;
    border-radius: 18px;
    box-shadow: 0 10px 26px rgba(16, 42, 67, 0.08);
    padding: 18px;
    margin-bottom: 18px;
}

.panel-title {
    font-size: 18px;
    font-weight: 800;
    color: #102a43;
    margin-bottom: 12px;
}

.controls-inline {
    display: grid;
    grid-template-columns: repeat(2, minmax(220px, 1fr));
    gap: 12px;
    margin-bottom: 14px;
}

.small-note {
    font-size: 12px;
    color: #66788a;
    margin-top: 8px;
}

@media (max-width: 1100px) {
    .filter-bar, .cards-grid, .controls-inline {
        grid-template-columns: 1fr 1fr;
    }
}

@media (max-width: 760px) {
    .filter-bar, .cards-grid, .controls-inline {
        grid-template-columns: 1fr;
    }

    .hero h1 {
        font-size: 24px;
    }
}
