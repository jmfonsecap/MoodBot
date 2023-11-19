import React, { ReactNode } from "react";
import styles from "../index.module.scss";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  children: ReactNode;
}

function MainModal({ isOpen, onClose, children }: Props) {
  return (
    isOpen && (
      <div className={styles.overlay}>
        <div className={styles.mainModal}>
          <div className={styles.modalContent}>{children}</div>
        </div>
      </div>
    )
  );
}
export default MainModal;
