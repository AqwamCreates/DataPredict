 document.addEventListener('DOMContentLoaded', (event) => {
	document.querySelectorAll('pre code').forEach((block) => {
		hljs.highlightBlock(block);
	});
});

function openSidebar() {
  document.getElementById("sidebar").style.width = "300px";
}

/* Set the width of the side navigation to 0 */
function closeSideBar() {
  document.getElementById("sidebar").style.width = "0";
}
