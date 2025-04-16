// JavaScript for handling the carousel functionality
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the carousel
    var carousel = document.querySelector('.carousel');
    var carouselItems = document.querySelectorAll('.carousel-item');
    var currentIndex = 0;

    function showSlide(index) {
        carouselItems.forEach((item, i) => {
            item.style.display = (i === index) ? 'block' : 'none';
        });
    }

    function nextSlide() {
        currentIndex = (currentIndex + 1) % carouselItems.length;
        showSlide(currentIndex);
    }

    // Show the first slide
    showSlide(currentIndex);

    // Set interval for automatic slide change
    setInterval(nextSlide, 5000); // Change slide every 5 seconds
});
