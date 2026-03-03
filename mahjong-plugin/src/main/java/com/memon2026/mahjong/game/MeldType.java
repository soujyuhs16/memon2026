package com.memon2026.mahjong.game;

/**
 * 面子类型枚举 (Meld type enum)
 */
public enum MeldType {
    PONG("碰"),           // 刻子（碰牌，3张同牌，明牌）
    KONG_OPEN("明杠"),    // 明杠（4张同牌，明牌）
    KONG_CLOSED("暗杠");  // 暗杠（4张同牌，暗牌）

    private final String display;

    MeldType(String display) {
        this.display = display;
    }

    public String getDisplay() { return display; }
}
