package com.memon2026.mahjong;

import com.memon2026.mahjong.game.Meld;
import com.memon2026.mahjong.game.MeldType;
import com.memon2026.mahjong.game.Tile;
import com.memon2026.mahjong.game.TileSuit;
import com.memon2026.mahjong.game.WinChecker;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * 广东麻将核心逻辑单元测试
 */
public class MahjongTest {

    // ---- Tile tests ----

    @Test
    public void testTileCreation() {
        Tile t = new Tile(TileSuit.WAN, 5);
        assertEquals(TileSuit.WAN, t.getSuit());
        assertEquals(5, t.getNumber());
        assertEquals("5万", t.toString());
    }

    @Test
    public void testTileInvalidNumber() {
        assertThrows(IllegalArgumentException.class, () -> new Tile(TileSuit.WAN, 0));
        assertThrows(IllegalArgumentException.class, () -> new Tile(TileSuit.WAN, 10));
    }

    @Test
    public void testTileEquality() {
        Tile a = new Tile(TileSuit.WAN, 3);
        Tile b = new Tile(TileSuit.WAN, 3);
        Tile c = new Tile(TileSuit.TIAO, 3);
        assertEquals(a, b);
        assertFalse(a.equals(c));
    }

    @Test
    public void testTileFromString() {
        assertEquals(new Tile(TileSuit.WAN, 1), Tile.fromString("1wan"));
        assertEquals(new Tile(TileSuit.WAN, 1), Tile.fromString("1w"));
        assertEquals(new Tile(TileSuit.WAN, 1), Tile.fromString("1万"));
        assertEquals(new Tile(TileSuit.TIAO, 9), Tile.fromString("9tiao"));
        assertEquals(new Tile(TileSuit.TIAO, 9), Tile.fromString("9t"));
        assertEquals(new Tile(TileSuit.TIAO, 9), Tile.fromString("9条"));
        assertEquals(new Tile(TileSuit.BING, 5), Tile.fromString("5bing"));
        assertEquals(new Tile(TileSuit.BING, 5), Tile.fromString("5b"));
        assertEquals(new Tile(TileSuit.BING, 5), Tile.fromString("5饼"));
    }

    @Test
    public void testTileFromStringInvalid() {
        assertThrows(IllegalArgumentException.class, () -> Tile.fromString("0wan"));
        assertThrows(IllegalArgumentException.class, () -> Tile.fromString("10wan"));
        assertThrows(IllegalArgumentException.class, () -> Tile.fromString("abc"));
        assertThrows(IllegalArgumentException.class, () -> Tile.fromString(""));
    }

    @Test
    public void testTileSort() {
        List<Tile> tiles = new ArrayList<>();
        tiles.add(new Tile(TileSuit.BING, 3));
        tiles.add(new Tile(TileSuit.WAN, 1));
        tiles.add(new Tile(TileSuit.TIAO, 5));
        Collections.sort(tiles);
        assertEquals(new Tile(TileSuit.WAN, 1), tiles.get(0));
        assertEquals(new Tile(TileSuit.TIAO, 5), tiles.get(1));
        assertEquals(new Tile(TileSuit.BING, 3), tiles.get(2));
    }

    // ---- WinChecker: toCounts ----

    @Test
    public void testToCounts() {
        List<Tile> tiles = Arrays.asList(
                new Tile(TileSuit.WAN, 1),
                new Tile(TileSuit.WAN, 1),
                new Tile(TileSuit.TIAO, 5)
        );
        int[] counts = WinChecker.toCounts(tiles);
        assertEquals(2, counts[0]);           // 1万 索引0
        assertEquals(1, counts[9 + 4]);       // 5条 索引13
        assertEquals(0, counts[18]);           // 1饼 索引18
    }

    // ---- WinChecker: queYiMen ----

    @Test
    public void testQueYiMen_missing_one() {
        // 全万+条，缺饼 → 满足缺一门
        List<Tile> hand = tiles(
                wan(1), wan(2), wan(3), wan(4), wan(5), wan(6),
                tiao(1), tiao(2), tiao(3), tiao(4), tiao(5),
                tiao(6), tiao(7), tiao(8)
        );
        assertTrue(WinChecker.queYiMen(hand, Collections.emptyList()));
    }

    @Test
    public void testQueYiMen_all_suits() {
        // 三种花色都有 → 不满足缺一门
        List<Tile> hand = tiles(wan(1), tiao(1), bing(1));
        assertFalse(WinChecker.queYiMen(hand, Collections.emptyList()));
    }

    @Test
    public void testQueYiMen_via_meld() {
        // 手牌全是万，但副子含饼 → 三种花色，不满足缺一门
        List<Tile> hand = tiles(wan(1), wan(2));
        List<Meld> melds = Collections.singletonList(
                new Meld(MeldType.PONG, Arrays.asList(bing(5), bing(5), bing(5)))
        );
        // 手牌有万，副子有饼，一共两种 → 满足缺一门（缺条）
        assertTrue(WinChecker.queYiMen(hand, melds));
    }

    // ---- WinChecker: canWin ----

    @Test
    public void testCanWin_basic_sequences() {
        // 4组顺子 + 1对将 (全万，缺条缺饼)
        // 123万 456万 789万 123万 + 55万 = 14张
        List<Tile> hand = tiles(
                wan(1), wan(2), wan(3),
                wan(4), wan(5), wan(6),
                wan(7), wan(8), wan(9),
                wan(1), wan(2), wan(3),
                wan(5), wan(5)
        );
        assertTrue(WinChecker.canWin(hand, Collections.emptyList()));
    }

    @Test
    public void testCanWin_basic_triplets() {
        // 4组刻子 + 1对将 (全万，缺条缺饼)
        // 111万 222万 333万 444万 + 55万 = 14张
        List<Tile> hand = tiles(
                wan(1), wan(1), wan(1),
                wan(2), wan(2), wan(2),
                wan(3), wan(3), wan(3),
                wan(4), wan(4), wan(4),
                wan(5), wan(5)
        );
        assertTrue(WinChecker.canWin(hand, Collections.emptyList()));
    }

    @Test
    public void testCanWin_with_pong_meld() {
        // 已有1个碰副子，手牌需要 3组面子 + 1对将 = 11张
        // 碰：333万
        // 手牌：123万 456万 789万 + 11万 = 11张
        List<Meld> melds = Collections.singletonList(
                new Meld(MeldType.PONG, Arrays.asList(wan(3), wan(3), wan(3)))
        );
        List<Tile> hand = tiles(
                wan(1), wan(2), wan(3),
                wan(4), wan(5), wan(6),
                wan(7), wan(8), wan(9),
                wan(1), wan(1)
        );
        assertTrue(WinChecker.canWin(hand, melds));
    }

    @Test
    public void testCanWin_fails_wrong_size() {
        // 13张，不满足14张要求（无副子时需14张）
        List<Tile> hand = tiles(
                wan(1), wan(2), wan(3),
                wan(4), wan(5), wan(6),
                wan(7), wan(8), wan(9),
                wan(1), wan(2), wan(3), wan(5)
        );
        assertFalse(WinChecker.canWin(hand, Collections.emptyList()));
    }

    @Test
    public void testCanWin_fails_que_yi_men() {
        // 满足牌型但三种花色都有 → 不能胡
        List<Tile> hand = tiles(
                wan(1), wan(2), wan(3),
                tiao(1), tiao(2), tiao(3),
                bing(1), bing(2), bing(3),
                wan(4), wan(5), wan(6),
                wan(7), wan(7)
        );
        assertFalse(WinChecker.canWin(hand, Collections.emptyList()));
    }

    @Test
    public void testCanWin_fails_incomplete() {
        // 14张但不能组成有效牌型
        List<Tile> hand = tiles(
                wan(1), wan(3), wan(5),
                wan(7), wan(9), wan(2),
                wan(4), wan(6), wan(8),
                wan(1), wan(3), wan(5), wan(7), wan(9)
        );
        assertFalse(WinChecker.canWin(hand, Collections.emptyList()));
    }

    // ---- WinChecker: isPengPengHu ----

    @Test
    public void testPengPengHu_true() {
        // 全刻子
        List<Tile> hand = tiles(
                wan(1), wan(1), wan(1),
                wan(2), wan(2), wan(2),
                wan(3), wan(3), wan(3),
                wan(4), wan(4), wan(4),
                wan(5), wan(5)
        );
        assertTrue(WinChecker.isPengPengHu(hand, Collections.emptyList()));
    }

    @Test
    public void testPengPengHu_false_has_sequence() {
        // 含顺子，不是碰碰胡
        List<Tile> hand = tiles(
                wan(1), wan(2), wan(3),
                wan(1), wan(1), wan(1),
                wan(4), wan(4), wan(4),
                wan(5), wan(5), wan(5),
                wan(6), wan(6)
        );
        assertFalse(WinChecker.isPengPengHu(hand, Collections.emptyList()));
    }

    // ---- WinChecker: isQingYiSe ----

    @Test
    public void testQingYiSe_true() {
        List<Tile> hand = tiles(wan(1), wan(2), wan(3), wan(4), wan(5));
        assertTrue(WinChecker.isQingYiSe(hand, Collections.emptyList()));
    }

    @Test
    public void testQingYiSe_false_mixed() {
        List<Tile> hand = tiles(wan(1), tiao(2));
        assertFalse(WinChecker.isQingYiSe(hand, Collections.emptyList()));
    }

    // ---- WinChecker: calculateFan ----

    @Test
    public void testCalculateFan_base() {
        // 万+条两种花色（缺饼），普通胡牌，基础1番
        List<Tile> hand = tiles(
                wan(1), wan(2), wan(3),
                wan(4), wan(5), wan(6),
                tiao(1), tiao(2), tiao(3),
                tiao(4), tiao(5), tiao(6),
                wan(9), wan(9)
        );
        assertEquals(1, WinChecker.calculateFan(hand, Collections.emptyList(), false));
    }

    @Test
    public void testCalculateFan_selfDraw() {
        // 万+条两种花色（缺饼），自摸，2番
        List<Tile> hand = tiles(
                wan(1), wan(2), wan(3),
                wan(4), wan(5), wan(6),
                tiao(1), tiao(2), tiao(3),
                tiao(4), tiao(5), tiao(6),
                wan(9), wan(9)
        );
        assertEquals(2, WinChecker.calculateFan(hand, Collections.emptyList(), true));
    }

    @Test
    public void testCalculateFan_qingYiSe() {
        List<Tile> hand = tiles(
                wan(1), wan(1), wan(1),
                wan(2), wan(2), wan(2),
                wan(3), wan(3), wan(3),
                wan(4), wan(4), wan(4),
                wan(5), wan(5)
        );
        // 清一色 +3，自摸 +1，基础 1 = 5
        assertEquals(5, WinChecker.calculateFan(hand, Collections.emptyList(), true));
    }

    @Test
    public void testCalculateFan_pengpenghu() {
        List<Tile> hand = tiles(
                wan(1), wan(1), wan(1),
                wan(2), wan(2), wan(2),
                wan(3), wan(3), wan(3),
                wan(4), wan(4), wan(4),
                wan(5), wan(5)
        );
        // 碰碰胡 +1，基础 1 = 2（不算清一色因为清一色已包含并取高）
        // 实际上清一色也满足，但 calculateFan 先检查清一色
        // 全万 = 清一色，所以是 +3
        assertEquals(4, WinChecker.calculateFan(hand, Collections.emptyList(), false));
    }

    // ---- Helper methods ----

    private static Tile wan(int n) { return new Tile(TileSuit.WAN, n); }
    private static Tile tiao(int n) { return new Tile(TileSuit.TIAO, n); }
    private static Tile bing(int n) { return new Tile(TileSuit.BING, n); }

    private static List<Tile> tiles(Tile... ts) {
        return new ArrayList<>(Arrays.asList(ts));
    }
}
