.ALTER SS_VDD_MAX_TEMP_MAX
.lib "../../device_models/libs/ptm130.spice.lib" SS
.TEMP 75
.PARAM vdd={VDD_MAX}

.ALTER SS_VDD_MAX_TEMP_MIN
.lib "../../device_models/libs/ptm130.spice.lib" SS
.TEMP 0
.PARAM vdd={VDD_MAX}

.ALTER SS_VDD_MIN_TEMP_MAX
.lib "../../device_models/libs/ptm130.spice.lib" SS
.TEMP 75
.PARAM vdd={VDD_MIN}

.ALTER SS_VDD_MIN_TEMP_MIN
.lib "../../device_models/libs/ptm130.spice.lib" SS
.TEMP 0
.PARAM vdd={VDD_MIN}



.ALTER FF_VDD_MAX_TEMP_MAX
.lib "../../device_models/libs/ptm130.spice.lib" FF
.TEMP 75
.PARAM vdd={VDD_MAX}

.ALTER FF_VDD_MAX_TEMP_MIN
.lib "../../device_models/libs/ptm130.spice.lib" FF
.TEMP 0
.PARAM vdd={VDD_MAX}

.ALTER FF_VDD_MIN_TEMP_MAX
.lib "../../device_models/libs/ptm130.spice.lib" FF
.TEMP 75
.PARAM vdd={VDD_MIN}

.ALTER FF_VDD_MIN_TEMP_MIN
.lib "../../device_models/libs/ptm130.spice.lib" FF
.TEMP 0
.PARAM vdd={VDD_MIN}


.ALTER SNFP_VDD_MAX_TEMP_MAX
.lib "../../device_models/libs/ptm130.spice.lib" SNFP
.TEMP 75
.PARAM vdd={VDD_MAX}

.ALTER SNFP_VDD_MAX_TEMP_MIN
.lib "../../device_models/libs/ptm130.spice.lib" SNFP
.TEMP 0
.PARAM vdd={VDD_MAX}

.ALTER SNFP_VDD_MIN_TEMP_MAX
.lib "../../device_models/libs/ptm130.spice.lib" SNFP
.TEMP 75
.PARAM vdd={VDD_MIN}

.ALTER SNFP_VDD_MIN_TEMP_MIN
.lib "../../device_models/libs/ptm130.spice.lib" SNFP
.TEMP 0
.PARAM vdd={VDD_MIN}



.ALTER FNSP_VDD_MAX_TEMP_MAX
.lib "../../device_models/libs/ptm130.spice.lib" FNSP
.TEMP 75
.PARAM vdd={VDD_MAX}

.ALTER FNSP_VDD_MAX_TEMP_MIN
.lib "../../device_models/libs/ptm130.spice.lib" FNSP
.TEMP 0
.PARAM vdd={VDD_MAX}

.ALTER FNSP_VDD_MIN_TEMP_MAX
.lib "../../device_models/libs/ptm130.spice.lib" FNSP
.TEMP 75
.PARAM vdd={VDD_MIN}

.ALTER FNSP_VDD_MIN_TEMP_MIN
.lib "../../device_models/libs/ptm130.spice.lib" FNSP
.TEMP 0
.PARAM vdd={VDD_MIN}
