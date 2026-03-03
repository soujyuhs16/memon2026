package com.memon2026.mahjong.game;

import java.util.Collections;
import java.util.List;

/**
 * 面子（副子）：碰牌或杠牌形成的明/暗牌组。
 * 广东麻将不允许吃牌（不能从他人弃牌组成顺子），
 * 所以面子只有碰（刻子）和杠（四张）两种。
 */
public class Meld {

    private final MeldType type;
    private final List<Tile> tiles;

    public Meld(MeldType type, List<Tile> tiles) {
        this.type = type;
        this.tiles = Collections.unmodifiableList(tiles);
    }

    public MeldType getType() { return type; }
    public List<Tile> getTiles() { return tiles; }

    /** 代表牌（取第一张）*/
    public Tile getRepTile() { return tiles.get(0); }

    public boolean isKong() {
        return type == MeldType.KONG_OPEN || type == MeldType.KONG_CLOSED;
    }

    /** 聊天显示格式 */
    public String getDisplay() {
        StringBuilder sb = new StringBuilder("[");
        sb.append(type.getDisplay()).append(":");
        for (Tile t : tiles) sb.append(t.getDisplay());
        sb.append("§r]");
        return sb.toString();
    }
}
