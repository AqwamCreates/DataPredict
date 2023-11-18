function openSidebar() {
  document.getElementById("sidebar").style.width = "300px";
}

/* Set the width of the side navigation to 0 */
function closeSideBar() {
  document.getElementById("sidebar").style.width = "0";
}

function scrollToTextWhenQueried() {
const urlParams = new URLSearchParams(window.location.search);
    const scrollToHeaderText = urlParams.get('scrollToHeaderText');
    // Scroll to the specified header
    if (scrollToHeaderText) {
        const headerElement = document.getElementById(scrollToHeaderText);
		alert(headerElement)
        if (headerElement) {
            headerElement.scrollIntoView({ behavior: 'smooth' });
        }
    }
}
