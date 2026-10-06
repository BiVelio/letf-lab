"""Ejecuta todas las líneas de investigación y regenera reports/."""

import importlib
import logging
import sys
import time

LINES = ["r1_decay", "r2_tracking", "r3_synthetic", "r4_close_flows",
         "r5_harvest", "r6_regime", "r7_liquidity", "r8_kelly", "r9_entry",
         "r10_predictability", "r11_doctoral",
         "r12_long_ml"]

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    only = sys.argv[1:] or LINES
    for name in only:
        t0 = time.time()
        importlib.import_module(f"letf_lab.research.{name}").run()
        print(f"   {name}: {time.time() - t0:.1f}s")
