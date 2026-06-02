document.addEventListener("DOMContentLoaded", () => {
  if (window.bulmaCarousel) {
    document.querySelectorAll(".carousel").forEach((el) => {
      bulmaCarousel.attach(el, {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        autoplay: false,
      });
    });
  }

  const switcher = document.getElementById("taskSwitcher");
  if (!switcher) return;

  function setActiveTask(task) {
    const t = task === "tomato" ? "tomato" : "cube";
    document.body.dataset.activeTask = t;
    switcher.querySelectorAll("li").forEach((li) => {
      li.classList.toggle("is-active", li.dataset.task === t);
    });
    document.querySelectorAll(".task-pane").forEach((pane) => {
      const vids = pane.querySelectorAll("video");
      if (pane.dataset.task === t) {
        vids.forEach((v) => {
          v.currentTime = 0;
          v.play().catch(() => {});
        });
      } else {
        vids.forEach((v) => v.pause());
      }
    });
    if (history.replaceState) {
      history.replaceState(null, "", "#" + t);
    }
  }

  switcher.querySelectorAll("li").forEach((li) => {
    li.addEventListener("click", (e) => {
      e.preventDefault();
      setActiveTask(li.dataset.task);
    });
  });

  const hash = (location.hash || "#cube").replace("#", "");
  setActiveTask(hash === "tomato" ? "tomato" : "cube");
});
