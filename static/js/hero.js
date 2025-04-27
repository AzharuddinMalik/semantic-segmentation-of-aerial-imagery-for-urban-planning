document.addEventListener('DOMContentLoaded', function() {
    const heroContainer = document.querySelector('.hero-image');
    const fallbackImages = [
        'https://images.unsplash.com/photo-1661961112835-ca6f5811d2af?ixlib=rb-1.2.1&auto=format&fit=crop&w=1000&q=80',
        'https://images.unsplash.com/photo-1569336415962-a4bd9f69c907?ixlib=rb-1.2.1&auto=format&fit=crop&w=1000&q=80'
    ];

    // Use fallback images directly
    initHeroCarousel(fallbackImages);

    function initHeroCarousel(images) {
        // Clear loading overlay if exists
        const overlay = heroContainer.querySelector('.hero-overlay');
        if (overlay) overlay.remove();

        // Add images to carousel
        images.forEach((imgUrl, index) => {
            const img = document.createElement('img');
            img.src = imgUrl;
            img.alt = `Urban view ${index + 1}`;
            img.classList.add(index === 0 ? 'active' : '');
            heroContainer.appendChild(img);
        });

        // Rotate images
        let currentIndex = 0;
        setInterval(() => {
            const images = heroContainer.querySelectorAll('img');
            images[currentIndex].classList.remove('active');
            currentIndex = (currentIndex + 1) % images.length;
            images[currentIndex].classList.add('active');
        }, 5000);
    }
});