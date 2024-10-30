export interface ZoomDragHTMLElement extends HTMLElement {
    /** Removes the behavior: hide on overflow of parent element */
    stopHiddenOverflowBehavior?(): ZoomDragHTMLElement | null;
    /** Removes the draggable behavior from element */
    stopDraggableBehavior?(): ZoomDragHTMLElement | null;
    /** Removes the zoomable behavior from element */
    stopZoomableBehavior?(): ZoomDragHTMLElement | null;
  }