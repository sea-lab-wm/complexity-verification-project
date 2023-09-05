// SNIPPET_STARTS
public HealthReport(int score, String iconUrl, Localizable description) {
    this.score = score;
    if (score <= 20) {
        this.iconClassName = HEALTH_0_TO_20;
    } else if (score <= 40) {
        this.iconClassName = HEALTH_21_TO_40;
    } else if (score <= 60) {
        this.iconClassName = HEALTH_41_TO_60;
    } else if (score <= 80) {
        this.iconClassName = HEALTH_61_TO_80;
    } else {
        this.iconClassName = HEALTH_OVER_80;
    }
    if (iconUrl == null) {
        if (score <= 20) {
            this.iconUrl = HEALTH_0_TO_20_IMG;
        } else if (score <= 40) {
            this.iconUrl = HEALTH_21_TO_40_IMG;
        } else if (score <= 60) {
            this.iconUrl = HEALTH_41_TO_60_IMG;
        } else if (score <= 80) {
            this.iconUrl = HEALTH_61_TO_80_IMG;
        } else {
            this.iconUrl = HEALTH_OVER_80_IMG;
        }
    } else {
        this.iconUrl = iconUrl;
    }
    this.description = null;
    setLocalizibleDescription(description);
}