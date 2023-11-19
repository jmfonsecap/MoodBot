import React from "react";
import { Routes } from "@config/routes";
import styles from "../index.module.scss";
function HeadBar() {
  return (
    <ul className={styles.headBar}>
      <li>
        <a href="/">Home</a>
      </li>
      <li>
        <a href="/products">Products</a>
      </li>
      <li>
        <a href="/documentation">Documentation</a>
      </li>
      <li>
        <a href="/pricing">Pricing</a>
      </li>
    </ul>
  );
}

export default HeadBar;
