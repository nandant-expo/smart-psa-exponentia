import { ZoomDragHTMLElement } from '.';

/**
 * Makes the Element hide when it overflows parent
 * @param param.element - The element to hide on overflow
 * @param param.CONTAINER_WITH_OVERFLOW_HIDDEN_CSS_CLASS - Custom CSS class name to be used (optional / internally provided)
 * @return parent of element with `.stopHiddenOverflowBehavior()` method to help stop behavior
 * @example
 * // make element hide when it overflows parent
 * const elmnt = hideElementOverflow({ element: document.getElementById('my-element') });
 * // now it's not hidden anymore
 * elmnt.stopHiddenOverflowBehavior();
 */
export const hideElementOverflow = ({
  element,
  CONTAINER_WITH_OVERFLOW_HIDDEN_CSS_CLASS = 'xyzw-container',
}: {
  element: HTMLElement;
  CONTAINER_WITH_OVERFLOW_HIDDEN_CSS_CLASS?: string;
}): ZoomDragHTMLElement | null => {
  // log error, if the element is not found, and exit execution
  if (!element) {
    console.error(`Error: hideElementOverflow() needs an element to work`);
    return null;
  }

  // check  if necessary CSS is already added to body (to avoid polluting html)
  const isTheRequiredCSSAlreadyInPlace = !!Array.from(
    element.ownerDocument.getElementsByTagName('style')
  ).filter((styleEl) =>
    styleEl?.textContent?.includes(
      `.${CONTAINER_WITH_OVERFLOW_HIDDEN_CSS_CLASS}`
    )
  )?.[0];

  // append the necessary CSS, to body, to enable hidding element when overflow occurs (if not appended already)
  if (!isTheRequiredCSSAlreadyInPlace) {
    element.ownerDocument.body.appendChild(
      Object.assign(element.ownerDocument.createElement('style'), {
        textContent: `
          .${CONTAINER_WITH_OVERFLOW_HIDDEN_CSS_CLASS} {
            position: relative !important;
            overflow: hidden !important;
          }
          `,
      })
    );
  }

  // add the necessary CSS class, to parent of the element, so that it hides
  // element when it overflows (if not added already)
  if (
    !element?.parentElement?.classList.contains(
      CONTAINER_WITH_OVERFLOW_HIDDEN_CSS_CLASS
    )
  ) {
    element?.parentElement?.classList.add(
      CONTAINER_WITH_OVERFLOW_HIDDEN_CSS_CLASS
    );
  }

  // returns an element with .stopHiddenOverflowBehavior() method to
  // help remove behavior from element, if needed
  return Object.assign(element, {
    stopHiddenOverflowBehavior() {
      element?.parentElement?.classList.remove(
        CONTAINER_WITH_OVERFLOW_HIDDEN_CSS_CLASS
      );
      return element;
    },
  });
};
