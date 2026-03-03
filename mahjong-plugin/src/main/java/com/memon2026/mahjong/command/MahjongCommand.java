package com.memon2026.mahjong.command;

import com.memon2026.mahjong.game.GamePlayer;
import com.memon2026.mahjong.game.GameState;
import com.memon2026.mahjong.game.MahjongGame;
import com.memon2026.mahjong.game.Tile;
import com.memon2026.mahjong.manager.GameManager;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.command.TabCompleter;
import org.bukkit.entity.Player;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/**
 * 麻将命令处理器
 * 命令: /mahjong (别名: /mj)
 *
 * 子命令：
 *   create <房间名>       - 创建房间
 *   join <房间名>         - 加入房间
 *   start                 - 开始游戏（房主）
 *   discard <牌序号/牌名> - 出牌
 *   peng                  - 碰牌
 *   gang                  - 杠牌（暗杠/明杠/补杠）
 *   hu                    - 胡牌
 *   pass                  - 放弃认领
 *   show                  - 显示手牌
 *   rooms                 - 列出所有房间
 *   leave                 - 离开房间
 */
public class MahjongCommand implements CommandExecutor, TabCompleter {

    private final GameManager gameManager;

    public MahjongCommand(GameManager gameManager) {
        this.gameManager = gameManager;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if (!(sender instanceof Player)) {
            sender.sendMessage("§c该命令只能由玩家使用");
            return true;
        }
        Player player = (Player) sender;

        if (args.length == 0) {
            sendHelp(player);
            return true;
        }

        String sub = args[0].toLowerCase();
        switch (sub) {
            case "create" -> handleCreate(player, args);
            case "join"   -> handleJoin(player, args);
            case "start"  -> handleStart(player);
            case "discard", "d" -> handleDiscard(player, args);
            case "peng", "p"    -> handlePeng(player);
            case "gang", "g"    -> handleGang(player);
            case "hu", "h"      -> handleHu(player);
            case "pass"         -> handlePass(player);
            case "show", "s"    -> handleShow(player);
            case "rooms", "list" -> handleRooms(player);
            case "leave"        -> handleLeave(player);
            default -> sendHelp(player);
        }
        return true;
    }

    // ---- Sub-command handlers ----

    private void handleCreate(Player player, String[] args) {
        if (args.length < 2) {
            player.sendMessage("§c用法: /mj create <房间名>");
            return;
        }
        String roomName = args[1];
        if (roomName.length() > 16) {
            player.sendMessage("§c房间名不能超过16个字符");
            return;
        }
        String err = gameManager.createRoom(roomName, player.getUniqueId(), player.getName());
        if (err != null) {
            player.sendMessage(err);
        } else {
            player.sendMessage("§a成功创建房间: §f" + roomName);
            player.sendMessage("§7等待其他玩家加入 §f(/mj join " + roomName + "§7)");
            player.sendMessage("§7准备好后开始: §f/mj start");
        }
    }

    private void handleJoin(Player player, String[] args) {
        if (args.length < 2) {
            player.sendMessage("§c用法: /mj join <房间名>");
            return;
        }
        String roomName = args[1];
        String err = gameManager.joinRoom(roomName, player.getUniqueId(), player.getName());
        if (err != null) {
            player.sendMessage(err);
        } else {
            MahjongGame game = gameManager.getGame(roomName);
            player.sendMessage("§a成功加入房间: §f" + roomName);
            player.sendMessage("§e当前玩家 " + game.getPlayerCount() + "/4");
            game.broadcastMessage("§e" + player.getName() + " 加入了房间！("
                    + game.getPlayerCount() + "/4)");
        }
    }

    private void handleStart(Player player) {
        MahjongGame game = gameManager.getPlayerGame(player.getUniqueId());
        if (game == null) {
            player.sendMessage("§c你不在任何房间中，请先 §f/mj create §c或 §f/mj join");
            return;
        }
        if (!game.getCreatorUuid().equals(player.getUniqueId())) {
            player.sendMessage("§c只有房主才能开始游戏");
            return;
        }
        String err = game.startGame();
        if (err != null) player.sendMessage(err);
    }

    private void handleDiscard(Player player, String[] args) {
        if (args.length < 2) {
            player.sendMessage("§c用法: /mj discard <序号> 或 <牌名如1wan/9tiao/5bing>");
            return;
        }
        MahjongGame game = gameManager.getPlayerGame(player.getUniqueId());
        if (game == null) { player.sendMessage("§c你不在任何房间中"); return; }
        if (game.getState() != GameState.PLAYING) { player.sendMessage("§c游戏未在进行中"); return; }

        GamePlayer gp = game.getGamePlayer(player.getUniqueId());
        if (gp == null) { player.sendMessage("§c你不在该游戏中"); return; }

        String tileArg = args[1];
        Tile tile = parseTile(player, gp, tileArg);
        if (tile == null) return;

        String err = game.discard(player.getUniqueId(), tile);
        if (err != null) player.sendMessage(err);
    }

    private void handlePeng(Player player) {
        MahjongGame game = gameManager.getPlayerGame(player.getUniqueId());
        if (game == null) { player.sendMessage("§c你不在任何房间中"); return; }
        String err = game.peng(player.getUniqueId());
        if (err != null) player.sendMessage(err);
    }

    private void handleGang(Player player) {
        MahjongGame game = gameManager.getPlayerGame(player.getUniqueId());
        if (game == null) { player.sendMessage("§c你不在任何房间中"); return; }
        String err = game.gang(player.getUniqueId());
        if (err != null) player.sendMessage(err);
    }

    private void handleHu(Player player) {
        MahjongGame game = gameManager.getPlayerGame(player.getUniqueId());
        if (game == null) { player.sendMessage("§c你不在任何房间中"); return; }
        String err = game.hu(player.getUniqueId());
        if (err != null) player.sendMessage(err);
    }

    private void handlePass(Player player) {
        MahjongGame game = gameManager.getPlayerGame(player.getUniqueId());
        if (game == null) { player.sendMessage("§c你不在任何房间中"); return; }
        String err = game.pass(player.getUniqueId());
        if (err != null) player.sendMessage(err);
    }

    private void handleShow(Player player) {
        MahjongGame game = gameManager.getPlayerGame(player.getUniqueId());
        if (game == null) { player.sendMessage("§c你不在任何房间中"); return; }
        if (game.getState() != GameState.PLAYING) { player.sendMessage("§c游戏未在进行中"); return; }
        GamePlayer gp = game.getGamePlayer(player.getUniqueId());
        if (gp != null) game.sendHandToPlayer(gp);
    }

    private void handleRooms(Player player) {
        Collection<MahjongGame> allGames = gameManager.getAllGames();
        if (allGames.isEmpty()) {
            player.sendMessage("§7当前没有任何房间，使用 §f/mj create <名称> §7创建");
            return;
        }
        player.sendMessage("§6======= 房间列表 =======");
        for (MahjongGame g : allGames) {
            String status = switch (g.getState()) {
                case WAITING -> "§a等待中";
                case PLAYING -> "§e游戏中";
                case ENDED   -> "§7已结束";
            };
            player.sendMessage("  §f" + g.getRoomName() + " §7(" + g.getPlayerCount()
                    + "/4) " + status + " §7牌墙:" + g.getWallSize());
        }
    }

    private void handleLeave(Player player) {
        String roomName = gameManager.getPlayerRoom(player.getUniqueId());
        if (roomName == null) {
            player.sendMessage("§c你不在任何房间中");
            return;
        }
        MahjongGame game = gameManager.getGame(roomName);
        if (game != null && game.getState() == GameState.PLAYING) {
            game.broadcastMessage("§c" + player.getName() + " 离开了游戏，游戏结束");
            game.broadcastMessage("§7房间 §f" + roomName + " §7已关闭");
            // 通知房间内其他玩家
            for (GamePlayer gp : game.getPlayers()) {
                gameManager.leaveRoom(gp.getUuid());
            }
        } else {
            String err = gameManager.leaveRoom(player.getUniqueId());
            if (err != null) { player.sendMessage(err); return; }
            if (game != null) {
                game.broadcastMessage("§e" + player.getName() + " 离开了房间");
            }
        }
        player.sendMessage("§a已离开房间 §f" + roomName);
    }

    // ---- Tile parsing helper ----

    /**
     * 解析出牌参数：支持序号（1-14）或牌名（1wan/9tiao/5bing 等）
     */
    private Tile parseTile(Player player, GamePlayer gp, String arg) {
        // 先尝试作为序号解析
        try {
            int idx = Integer.parseInt(arg);
            Tile t = gp.getTileByIndex(idx);
            if (t == null) {
                player.sendMessage("§c无效的序号: §f" + arg
                        + " §c(有效范围 1-" + gp.getHand().size() + ")");
                return null;
            }
            return t;
        } catch (NumberFormatException ignored) {
            // 不是数字，继续尝试牌名解析
        }
        // 尝试作为牌名解析
        try {
            return Tile.fromString(arg);
        } catch (IllegalArgumentException e) {
            player.sendMessage("§c无法识别的牌: §f" + arg);
            player.sendMessage("§7格式：序号(1-" + gp.getHand().size()
                    + ") 或牌名(1wan/1w/1万, 9tiao/9t/9条, 5bing/5b/5饼)");
            return null;
        }
    }

    // ---- Help ----

    private void sendHelp(Player player) {
        player.sendMessage("§6======= 广东麻将 帮助 =======");
        player.sendMessage("§e/mj create <房间名> §7- 创建房间");
        player.sendMessage("§e/mj join <房间名>   §7- 加入房间");
        player.sendMessage("§e/mj start           §7- 开始游戏（房主）");
        player.sendMessage("§e/mj discard <牌>    §7- 出牌（序号或牌名如1wan/9t/5bing）");
        player.sendMessage("§e/mj peng            §7- 碰牌");
        player.sendMessage("§e/mj gang            §7- 杠牌（暗杠/明杠/补杠）");
        player.sendMessage("§e/mj hu              §7- 胡牌（自摸或放炮）");
        player.sendMessage("§e/mj pass            §7- 放弃碰/杠/胡机会");
        player.sendMessage("§e/mj show            §7- 显示手牌");
        player.sendMessage("§e/mj rooms           §7- 列出所有房间");
        player.sendMessage("§e/mj leave           §7- 离开当前房间");
        player.sendMessage("§7别名: §f/mj§7 等同于 §f/mahjong");
    }

    // ---- Tab completion ----

    @Override
    public List<String> onTabComplete(CommandSender sender, Command command, String alias, String[] args) {
        List<String> completions = new ArrayList<>();
        if (args.length == 1) {
            List<String> subs = Arrays.asList("create", "join", "start", "discard",
                    "peng", "gang", "hu", "pass", "show", "rooms", "leave");
            String partial = args[0].toLowerCase();
            for (String s : subs) {
                if (s.startsWith(partial)) completions.add(s);
            }
        } else if (args.length == 2 && args[0].equalsIgnoreCase("join")) {
            // 列出可加入的房间
            String partial = args[1].toLowerCase();
            for (MahjongGame g : gameManager.getAllGames()) {
                if (g.getState() == GameState.WAITING && !g.isFull()) {
                    if (g.getRoomName().toLowerCase().startsWith(partial)) {
                        completions.add(g.getRoomName());
                    }
                }
            }
        }
        return completions;
    }
}
