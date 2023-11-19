import React, { ReactNode } from "react";
import styles from "../index.module.scss";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  children: ReactNode;
}

function Modal({ isOpen, onClose, children }: Props) {
  return (
    isOpen && (
      <div className={styles.overlay}>
        <div className={styles.modal}>
          <div className={styles.modalContent}>{children}</div>
        </div>
      </div>
    )
  );
}

export default Modal;
