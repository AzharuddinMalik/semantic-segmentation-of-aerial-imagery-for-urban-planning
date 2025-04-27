// Add at the very top
let areaChartInstance = null;
let coverageChartInstance = null;
let sliderInitialized = false;

$(document).ready(function() {

    /// Update switchTab function
    function switchTab(tabId) {
    $(".tab-content > div.active").fadeOut(200, function() {
        $(this).removeClass("active");
        $("#" + tabId + "-tab").fadeIn(200).addClass("active");

        // Handle tab-specific initializations
        handleTabSwitch(tabId);  // Moved inside completion callback
    });

    // Update tab buttons
    $(".tab-button").removeClass("border-blue-600 text-blue-600")
        .addClass("border-transparent text-gray-500 hover:text-gray-700");
    $(`[data-tab="${tabId}"]`).removeClass("border-transparent text-gray-500 hover:text-gray-700")
        .addClass("border-blue-600 text-blue-600");
}

    $(".tab-button").click(function() {
        const tabId = $(this).data("tab");
        switchTab(tabId);
    });



function initSlider() {
    if(!sliderInitialized && $(".twentytwenty-container img").length === 2) {
        $(".twentytwenty-container").twentytwenty({
            default_offset_pct: 0.5,
            orientation: 'horizontal',
            before_label: 'Original',
            after_label: 'Segmented',
            no_overlay: false,
            move_slider_on_hover: false,
            move_with_handle_only: true,
            click_to_move: true
        });
        sliderInitialized = true;
    }
}
    initSlider();
    $(window).on('resize', function() {
        $(".twentytwenty-container").twentytwenty('refresh');
    });
// Image load handler with error checking
$(".twentytwenty-container img").on('load', function() {
    if($(".twentytwenty-container img").length === 2) {
        initSlider();
        $(".twentytwenty-container").twentytwenty('refresh');
    }
}).on('error', function() {
    console.error('Image failed to load:', this.src);
});

    // --- Chart and Stats Initialization (Lazy) ---
    // --- Chart and Stats Initialization (Lazy) ---
function initializeCharts() {
    if(!window.segmentationData || !window.segmentationData.area_distribution) {
        console.error('Segmentation data not found!');
        return;
    }

    // Destroy existing charts
    if(areaChartInstance) areaChartInstance.destroy();
    if(coverageChartInstance) coverageChartInstance.destroy();

    // Safeguard data length
    const safeData = {
        area: window.segmentationData.area_distribution.slice(0, 6),
        coverage: window.segmentationData.class_coverage.slice(0, 6)
    };

    // Area Distribution Pie Chart
    areaChartInstance = new Chart(document.getElementById('areaChart'), {
        type: 'pie',
        data: {
            labels: ["Water", "Vegetation", "Road", "Building", "Land", "Unlabeled"],
            datasets: [{
                data: window.segmentationData.area_distribution,
                backgroundColor: CLASS_COLORS,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        boxWidth: 12,
                        padding: 20
                    }
                }
            }
        }
    });

    // Class Coverage Bar Chart
    coverageChartInstance = new Chart(document.getElementById('coverageChart'), {
        type: 'bar',
        data: {
            labels: ["Water", "Vegetation", "Road", "Building", "Land", "Unlabeled"],
            datasets: [{
                label: 'Coverage Percentage',
                data: window.segmentationData.class_coverage,
                backgroundColor: 'rgba(59, 130, 246, 0.5)',
                borderColor: 'rgba(59, 130, 246, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Percentage'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Update populateStats function
function populateStats() {
    const statsContainer = document.getElementById('stats-container');
    const statsData = window.segmentationData.stats;

    const stats = [
        { name: "Total Area", value: statsData.total_area },
        { name: "Buildings Density", value: statsData.building_density },
        { name: "Green Space Ratio", value: statsData.green_space },
        { name: "Water Coverage", value: statsData.water_coverage },
        { name: "Road Network Density", value: statsData.road_density }
    ];

    stats.forEach(stat => {
        const statElement = document.createElement('div');
        statElement.className = 'flex justify-between py-2 border-b';
        statElement.innerHTML = `
            <span class="font-medium text-gray-700">${stat.name}</span>
            <span class="font-semibold text-blue-600">${stat.value}</span>
        `;
        statsContainer.appendChild(statElement);
    });
}
function handleTabSwitch(tabId) {
    if(tabId === "analysis") {
        if(!window.analysisInitialized) {
            initializeCharts();
            populateStats();
            window.analysisInitialized = true;
        }
    }
    else if(tabId === "visualization") {
        initSlider();
        $(".twentytwenty-container").twentytwenty('refresh');
    }

    // Cleanup when leaving analysis tab
    if(tabId !== "analysis" && window.analysisInitialized) {
        if(areaChartInstance) {
            areaChartInstance.destroy();
            areaChartInstance = null;
        }
        if(coverageChartInstance) {
            coverageChartInstance.destroy();
            coverageChartInstance = null;
        }
        window.analysisInitialized = false;
    }
}
    // --- Default Active Tab ---
    switchTab("visualization");

});
