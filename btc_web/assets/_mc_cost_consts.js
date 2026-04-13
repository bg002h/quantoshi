/* Static constants consumed by the 5 MC-cost clientside callbacks in
   btc_web/callbacks/mc_controls.py. Hoisted out of the callback strings to
   avoid 5x duplication in /_dash-dependencies.
   REGEN: bump any of FREQ_PPY / _MC_PRICE_LIVE / mc_cache / btcpay / colors
   constants used below, then re-run the one-liner at the top of
   btc_web/callbacks/mc_controls.py (_MC_COST_CONSTS section) to get a fresh
   JSON blob and paste it here. */
window.QS_MC_COST_CONSTS = {"FREQ_PPY":{"Daily":365,"Weekly":52,"Monthly":12,"Quarterly":4,"Annually":1},"PRICE_LIVE":{"10":500,"20":1000,"30":1500,"40":2000},"BASE_SIMS":800,"BASE_PPY":12,"BASE_BINS":5,"MC_BINS":5,"MC_FREE_SIMS":200,"MC_FREQ":"Monthly","MC_DEFAULT_ENTRY_Q":10,"CACHED_MODEL_KEYS":["bub","ef","exp","lppl","pl","qr"],"CACHED_START_YRS":[2028,2031,2035],"MC_YEARS_OPTIONS":[40],"ENTRY_PCT_BINS":[0.01,0.1,0.5],"MC_FREE_GREEN":"#1a8f3c","MC_LIVE_AMBER":"#c57600","DIM_TEXT":"#555555","FALLBACK_MODEL_GRAY":"#888888","KNIGHT_GOLD":"#b8860b","UI_FONT_SM":"10px","UI_FONT_LG":"13px"};
