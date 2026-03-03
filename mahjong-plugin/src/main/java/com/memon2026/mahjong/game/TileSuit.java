package com.memon2026.mahjong.game;

/**
 * 麻将牌花色枚举 (Mahjong tile suit enum)
 * 广东麻将使用三种花色：万、条、饼
 */
public enum TileSuit {
    WAN("万", "§e"),   // 万子 (Characters) - Yellow
    TIAO("条", "§a"),  // 条子 (Bamboo) - Green
    BING("饼", "§b");  // 饼子 (Circles) - Aqua

    private final String display;
    private final String color;

    TileSuit(String display, String color) {
        this.display = display;
        this.color = color;
    }

    /** 显示名称 */
    public String getDisplay() { return display; }

    /** Chat颜色代码 */
    public String getColor() { return color; }
}
