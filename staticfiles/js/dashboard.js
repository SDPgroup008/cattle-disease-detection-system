function initializeCharts(dashboardData) {
    // Default values if data is missing
    const cattleCount = dashboardData.cattle_count || 0;
    const healthyCount = dashboardData.healthy_count || 0;
    const infectedCount = dashboardData.infected_count || 0;
    const footInfections = dashboardData.foot_infections || 0;
    const mouthInfections = dashboardData.mouth_infections || 0;
    const trendDates = dashboardData.trend_dates || [];
    const trendHealthy = dashboardData.trend_healthy || [0];
    const trendFoot = dashboardData.trend_foot || [0];
    const trendMouth = dashboardData.trend_mouth || [0];

    console.log('Chart Data:', { trendDates, trendHealthy, trendFoot, trendMouth });

    try {
        // Pie Chart: Healthy vs Infected
        const ctxPie = document.getElementById('healthPieChart').getContext('2d');
        new Chart(ctxPie, {
            type: 'pie',
            data: {
                labels: ['Healthy', 'Infected'],
                datasets: [{
                    label: 'Cattle Health',
                    data: [healthyCount, infectedCount],
                    backgroundColor: ['#28a745', '#dc3545'],
                    borderColor: ['#ffffff', '#ffffff'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'top', labels: { color: '#ffffff' } },
                    title: { display: true, text: 'Healthy vs Infected Cattle', color: '#ffffff' }
                }
            }
        });

        // Bar Chart: Foot vs Mouth Infections
        const ctxBar = document.getElementById('infectionBarChart').getContext('2d');
        new Chart(ctxBar, {
            type: 'bar',
            data: {
                labels: ['Foot Infection', 'Mouth Infection'],
                datasets: [{
                    label: 'Infection Count',
                    data: [footInfections, mouthInfections],
                    backgroundColor: ['#ff7875', '#ffd666'],
                    borderColor: ['#d46b68', '#d4b013'],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: { y: { beginAtZero: true, title: { display: true, text: 'Count', color: '#ffffff' } } },
                plugins: {
                    legend: { display: true, position: 'top', labels: { color: '#ffffff' } },
                    title: { display: true, text: 'Foot vs Mouth Infections', color: '#ffffff' }
                }
            }
        });

        // Line Chart: Health Trend Over Time
        const ctxLine = document.getElementById('healthTrendChart').getContext('2d');
        new Chart(ctxLine, {
            type: 'line',
            data: {
                labels: trendDates.length ? trendDates : ['No Data'],
                datasets: [
                    { label: 'Healthy', data: trendHealthy, borderColor: '#28a745', backgroundColor: 'rgba(40, 167, 69, 0.2)', fill: true, borderWidth: 2 },
                    { label: 'Foot Infected', data: trendFoot, borderColor: '#ff7875', backgroundColor: 'rgba(255, 120, 117, 0.2)', fill: true, borderWidth: 2 },
                    { label: 'Mouth Infected', data: trendMouth, borderColor: '#ffd666', backgroundColor: 'rgba(255, 214, 102, 0.2)', fill: true, borderWidth: 2 }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    x: { type: 'category', title: { display: true, text: 'Date', color: '#ffffff' } },
                    y: { beginAtZero: true, title: { display: true, text: 'Count', color: '#ffffff' } }
                },
                plugins: {
                    legend: { display: true, position: 'top', labels: { color: '#ffffff' } },
                    title: { display: true, text: 'Health Trend Over Time', color: '#ffffff' }
                }
            }
        });
    } catch (chartError) {
        console.error('Chart initialization error:', chartError);
    }
}