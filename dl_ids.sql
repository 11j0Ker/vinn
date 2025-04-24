/*
 Navicat Premium Data Transfer

 Source Server         : 1
 Source Server Type    : MySQL
 Source Server Version : 80025
 Source Host           : localhost:3306
 Source Schema         : dl_ids

 Target Server Type    : MySQL
 Target Server Version : 80025
 File Encoding         : 65001

 Date: 17/02/2025 23:14:40
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for auth_group
-- ----------------------------
DROP TABLE IF EXISTS `auth_group`;
CREATE TABLE `auth_group`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(150) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `name`(`name`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of auth_group
-- ----------------------------

-- ----------------------------
-- Table structure for auth_group_permissions
-- ----------------------------
DROP TABLE IF EXISTS `auth_group_permissions`;
CREATE TABLE `auth_group_permissions`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `group_id` int NOT NULL,
  `permission_id` int NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `auth_group_permissions_group_id_permission_id_0cd325b0_uniq`(`group_id`, `permission_id`) USING BTREE,
  INDEX `auth_group_permissio_permission_id_84c5c92e_fk_auth_perm`(`permission_id`) USING BTREE,
  CONSTRAINT `auth_group_permissio_permission_id_84c5c92e_fk_auth_perm` FOREIGN KEY (`permission_id`) REFERENCES `auth_permission` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `auth_group_permissions_group_id_b120cbf9_fk_auth_group_id` FOREIGN KEY (`group_id`) REFERENCES `auth_group` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of auth_group_permissions
-- ----------------------------

-- ----------------------------
-- Table structure for auth_permission
-- ----------------------------
DROP TABLE IF EXISTS `auth_permission`;
CREATE TABLE `auth_permission`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `content_type_id` int NOT NULL,
  `codename` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `auth_permission_content_type_id_codename_01ab375a_uniq`(`content_type_id`, `codename`) USING BTREE,
  CONSTRAINT `auth_permission_content_type_id_2f476e4b_fk_django_co` FOREIGN KEY (`content_type_id`) REFERENCES `django_content_type` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 45 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of auth_permission
-- ----------------------------
INSERT INTO `auth_permission` VALUES (1, 'Can add log entry', 1, 'add_logentry');
INSERT INTO `auth_permission` VALUES (2, 'Can change log entry', 1, 'change_logentry');
INSERT INTO `auth_permission` VALUES (3, 'Can delete log entry', 1, 'delete_logentry');
INSERT INTO `auth_permission` VALUES (4, 'Can view log entry', 1, 'view_logentry');
INSERT INTO `auth_permission` VALUES (5, 'Can add permission', 2, 'add_permission');
INSERT INTO `auth_permission` VALUES (6, 'Can change permission', 2, 'change_permission');
INSERT INTO `auth_permission` VALUES (7, 'Can delete permission', 2, 'delete_permission');
INSERT INTO `auth_permission` VALUES (8, 'Can view permission', 2, 'view_permission');
INSERT INTO `auth_permission` VALUES (9, 'Can add group', 3, 'add_group');
INSERT INTO `auth_permission` VALUES (10, 'Can change group', 3, 'change_group');
INSERT INTO `auth_permission` VALUES (11, 'Can delete group', 3, 'delete_group');
INSERT INTO `auth_permission` VALUES (12, 'Can view group', 3, 'view_group');
INSERT INTO `auth_permission` VALUES (13, 'Can add user', 4, 'add_user');
INSERT INTO `auth_permission` VALUES (14, 'Can change user', 4, 'change_user');
INSERT INTO `auth_permission` VALUES (15, 'Can delete user', 4, 'delete_user');
INSERT INTO `auth_permission` VALUES (16, 'Can view user', 4, 'view_user');
INSERT INTO `auth_permission` VALUES (17, 'Can add content type', 5, 'add_contenttype');
INSERT INTO `auth_permission` VALUES (18, 'Can change content type', 5, 'change_contenttype');
INSERT INTO `auth_permission` VALUES (19, 'Can delete content type', 5, 'delete_contenttype');
INSERT INTO `auth_permission` VALUES (20, 'Can view content type', 5, 'view_contenttype');
INSERT INTO `auth_permission` VALUES (21, 'Can add session', 6, 'add_session');
INSERT INTO `auth_permission` VALUES (22, 'Can change session', 6, 'change_session');
INSERT INTO `auth_permission` VALUES (23, 'Can delete session', 6, 'delete_session');
INSERT INTO `auth_permission` VALUES (24, 'Can view session', 6, 'view_session');
INSERT INTO `auth_permission` VALUES (25, 'Can add user', 7, 'add_user');
INSERT INTO `auth_permission` VALUES (26, 'Can change user', 7, 'change_user');
INSERT INTO `auth_permission` VALUES (27, 'Can delete user', 7, 'delete_user');
INSERT INTO `auth_permission` VALUES (28, 'Can view user', 7, 'view_user');
INSERT INTO `auth_permission` VALUES (29, 'Can add task', 8, 'add_task');
INSERT INTO `auth_permission` VALUES (30, 'Can change task', 8, 'change_task');
INSERT INTO `auth_permission` VALUES (31, 'Can delete task', 8, 'delete_task');
INSERT INTO `auth_permission` VALUES (32, 'Can view task', 8, 'view_task');
INSERT INTO `auth_permission` VALUES (33, 'Can add tuning models', 9, 'add_tuningmodels');
INSERT INTO `auth_permission` VALUES (34, 'Can change tuning models', 9, 'change_tuningmodels');
INSERT INTO `auth_permission` VALUES (35, 'Can delete tuning models', 9, 'delete_tuningmodels');
INSERT INTO `auth_permission` VALUES (36, 'Can view tuning models', 9, 'view_tuningmodels');
INSERT INTO `auth_permission` VALUES (37, 'Can add ip address rule', 10, 'add_ipaddressrule');
INSERT INTO `auth_permission` VALUES (38, 'Can change ip address rule', 10, 'change_ipaddressrule');
INSERT INTO `auth_permission` VALUES (39, 'Can delete ip address rule', 10, 'delete_ipaddressrule');
INSERT INTO `auth_permission` VALUES (40, 'Can view ip address rule', 10, 'view_ipaddressrule');
INSERT INTO `auth_permission` VALUES (41, 'Can add captcha store', 11, 'add_captchastore');
INSERT INTO `auth_permission` VALUES (42, 'Can change captcha store', 11, 'change_captchastore');
INSERT INTO `auth_permission` VALUES (43, 'Can delete captcha store', 11, 'delete_captchastore');
INSERT INTO `auth_permission` VALUES (44, 'Can view captcha store', 11, 'view_captchastore');

-- ----------------------------
-- Table structure for auth_user
-- ----------------------------
DROP TABLE IF EXISTS `auth_user`;
CREATE TABLE `auth_user`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `password` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `last_login` datetime(6) NULL DEFAULT NULL,
  `is_superuser` tinyint(1) NOT NULL,
  `username` varchar(150) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `first_name` varchar(150) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `last_name` varchar(150) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `email` varchar(254) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `is_staff` tinyint(1) NOT NULL,
  `is_active` tinyint(1) NOT NULL,
  `date_joined` datetime(6) NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `username`(`username`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of auth_user
-- ----------------------------

-- ----------------------------
-- Table structure for auth_user_groups
-- ----------------------------
DROP TABLE IF EXISTS `auth_user_groups`;
CREATE TABLE `auth_user_groups`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `group_id` int NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `auth_user_groups_user_id_group_id_94350c0c_uniq`(`user_id`, `group_id`) USING BTREE,
  INDEX `auth_user_groups_group_id_97559544_fk_auth_group_id`(`group_id`) USING BTREE,
  CONSTRAINT `auth_user_groups_group_id_97559544_fk_auth_group_id` FOREIGN KEY (`group_id`) REFERENCES `auth_group` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `auth_user_groups_user_id_6a12ed8b_fk_auth_user_id` FOREIGN KEY (`user_id`) REFERENCES `auth_user` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of auth_user_groups
-- ----------------------------

-- ----------------------------
-- Table structure for auth_user_user_permissions
-- ----------------------------
DROP TABLE IF EXISTS `auth_user_user_permissions`;
CREATE TABLE `auth_user_user_permissions`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `user_id` int NOT NULL,
  `permission_id` int NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `auth_user_user_permissions_user_id_permission_id_14a6b632_uniq`(`user_id`, `permission_id`) USING BTREE,
  INDEX `auth_user_user_permi_permission_id_1fbb5f2c_fk_auth_perm`(`permission_id`) USING BTREE,
  CONSTRAINT `auth_user_user_permi_permission_id_1fbb5f2c_fk_auth_perm` FOREIGN KEY (`permission_id`) REFERENCES `auth_permission` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `auth_user_user_permissions_user_id_a95ead1b_fk_auth_user_id` FOREIGN KEY (`user_id`) REFERENCES `auth_user` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of auth_user_user_permissions
-- ----------------------------

-- ----------------------------
-- Table structure for captcha_captchastore
-- ----------------------------
DROP TABLE IF EXISTS `captcha_captchastore`;
CREATE TABLE `captcha_captchastore`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `challenge` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `response` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `hashkey` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `expiration` datetime(6) NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `hashkey`(`hashkey`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 24 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of captcha_captchastore
-- ----------------------------
INSERT INTO `captcha_captchastore` VALUES (22, 'IQLR', 'iqlr', '07cf923e45e6efed77b45c3208ad170efaf2b257', '2025-02-16 13:28:11.623620');

-- ----------------------------
-- Table structure for django_admin_log
-- ----------------------------
DROP TABLE IF EXISTS `django_admin_log`;
CREATE TABLE `django_admin_log`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `action_time` datetime(6) NOT NULL,
  `object_id` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL,
  `object_repr` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `action_flag` smallint UNSIGNED NOT NULL,
  `change_message` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `content_type_id` int NULL DEFAULT NULL,
  `user_id` int NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `django_admin_log_content_type_id_c4bce8eb_fk_django_co`(`content_type_id`) USING BTREE,
  INDEX `django_admin_log_user_id_c564eba6_fk_auth_user_id`(`user_id`) USING BTREE,
  CONSTRAINT `django_admin_log_content_type_id_c4bce8eb_fk_django_co` FOREIGN KEY (`content_type_id`) REFERENCES `django_content_type` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `django_admin_log_user_id_c564eba6_fk_auth_user_id` FOREIGN KEY (`user_id`) REFERENCES `auth_user` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of django_admin_log
-- ----------------------------

-- ----------------------------
-- Table structure for django_content_type
-- ----------------------------
DROP TABLE IF EXISTS `django_content_type`;
CREATE TABLE `django_content_type`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `app_label` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `model` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `django_content_type_app_label_model_76bd3d3b_uniq`(`app_label`, `model`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 12 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of django_content_type
-- ----------------------------
INSERT INTO `django_content_type` VALUES (1, 'admin', 'logentry');
INSERT INTO `django_content_type` VALUES (3, 'auth', 'group');
INSERT INTO `django_content_type` VALUES (2, 'auth', 'permission');
INSERT INTO `django_content_type` VALUES (4, 'auth', 'user');
INSERT INTO `django_content_type` VALUES (11, 'captcha', 'captchastore');
INSERT INTO `django_content_type` VALUES (5, 'contenttypes', 'contenttype');
INSERT INTO `django_content_type` VALUES (10, 'main', 'ipaddressrule');
INSERT INTO `django_content_type` VALUES (8, 'main', 'task');
INSERT INTO `django_content_type` VALUES (9, 'main', 'tuningmodels');
INSERT INTO `django_content_type` VALUES (7, 'main', 'user');
INSERT INTO `django_content_type` VALUES (6, 'sessions', 'session');

-- ----------------------------
-- Table structure for django_migrations
-- ----------------------------
DROP TABLE IF EXISTS `django_migrations`;
CREATE TABLE `django_migrations`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `app` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `applied` datetime(6) NOT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 24 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of django_migrations
-- ----------------------------
INSERT INTO `django_migrations` VALUES (1, 'contenttypes', '0001_initial', '2025-02-10 14:00:27.386979');
INSERT INTO `django_migrations` VALUES (2, 'auth', '0001_initial', '2025-02-10 14:00:28.227119');
INSERT INTO `django_migrations` VALUES (3, 'admin', '0001_initial', '2025-02-10 14:00:28.372436');
INSERT INTO `django_migrations` VALUES (4, 'admin', '0002_logentry_remove_auto_add', '2025-02-10 14:00:28.379199');
INSERT INTO `django_migrations` VALUES (5, 'admin', '0003_logentry_add_action_flag_choices', '2025-02-10 14:00:28.386680');
INSERT INTO `django_migrations` VALUES (6, 'contenttypes', '0002_remove_content_type_name', '2025-02-10 14:00:28.498117');
INSERT INTO `django_migrations` VALUES (7, 'auth', '0002_alter_permission_name_max_length', '2025-02-10 14:00:28.590021');
INSERT INTO `django_migrations` VALUES (8, 'auth', '0003_alter_user_email_max_length', '2025-02-10 14:00:28.637591');
INSERT INTO `django_migrations` VALUES (9, 'auth', '0004_alter_user_username_opts', '2025-02-10 14:00:28.644831');
INSERT INTO `django_migrations` VALUES (10, 'auth', '0005_alter_user_last_login_null', '2025-02-10 14:00:28.752656');
INSERT INTO `django_migrations` VALUES (11, 'auth', '0006_require_contenttypes_0002', '2025-02-10 14:00:28.756042');
INSERT INTO `django_migrations` VALUES (12, 'auth', '0007_alter_validators_add_error_messages', '2025-02-10 14:00:28.763420');
INSERT INTO `django_migrations` VALUES (13, 'auth', '0008_alter_user_username_max_length', '2025-02-10 14:00:28.827393');
INSERT INTO `django_migrations` VALUES (14, 'auth', '0009_alter_user_last_name_max_length', '2025-02-10 14:00:28.904480');
INSERT INTO `django_migrations` VALUES (15, 'auth', '0010_alter_group_name_max_length', '2025-02-10 14:00:28.922296');
INSERT INTO `django_migrations` VALUES (16, 'auth', '0011_update_proxy_permissions', '2025-02-10 14:00:28.934263');
INSERT INTO `django_migrations` VALUES (17, 'auth', '0012_alter_user_first_name_max_length', '2025-02-10 14:00:28.996764');
INSERT INTO `django_migrations` VALUES (18, 'captcha', '0001_initial', '2025-02-10 14:00:29.030388');
INSERT INTO `django_migrations` VALUES (19, 'captcha', '0002_alter_captchastore_id', '2025-02-10 14:00:29.034372');
INSERT INTO `django_migrations` VALUES (20, 'sessions', '0001_initial', '2025-02-10 14:00:29.068359');
INSERT INTO `django_migrations` VALUES (21, 'main', '0001_initial', '2025-02-10 14:05:29.796015');
INSERT INTO `django_migrations` VALUES (22, 'main', '0002_tuningmodels', '2025-02-10 14:05:29.879783');
INSERT INTO `django_migrations` VALUES (23, 'main', '0003_alter_tuningmodels_alpha_and_more', '2025-02-10 14:05:29.886632');

-- ----------------------------
-- Table structure for django_session
-- ----------------------------
DROP TABLE IF EXISTS `django_session`;
CREATE TABLE `django_session`  (
  `session_key` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `session_data` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `expire_date` datetime(6) NOT NULL,
  PRIMARY KEY (`session_key`) USING BTREE,
  INDEX `django_session_expire_date_a5c62663`(`expire_date`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of django_session
-- ----------------------------
INSERT INTO `django_session` VALUES ('3913sqgialvic254qpp8jk54dokqekuw', '.eJyFj01uhDAMha9SeU2kOAQIuUxkEqegoTDCQSN1NHdvGHXRXZff-5H9nrBIWPfPZQNfjpMbeEM4hQ_wT1gSeGzgwo2-GDwUlgINbEu8_SqV7iTy2I8UZpK5SkNEN-Z-wn7EnKY28zikOGROlqfRxL8VobXUCratdaYaUqic8j4bD6bCgS7faNMpbRTqD7Red77DGj7v6Z9E3Udnmfdj-a7PZlqFXw0Uklu4xsHALlsck8otamUTOuVitIqz7lzqjU00wesH4n5ZHA:1thXdO:3RW8_QyhP-U3pdgNUEYZplZISPU_edisk3BY8P7BjGs', '2025-02-24 17:31:50.381608');
INSERT INTO `django_session` VALUES ('6sjdwqmt38965w87z8jnibtzahv7nxh6', '.eJyFj8sKgzAQRX-lzFrBxHd-RsZkUkOtSiahUPHfG6WL7ro898HM3cHxMK93t4AKPlIGFwyRyYPawRlQIoMTF3wSKAjEATJYnH58lUQbMr9Wb4YJeUpSq0XX22YUTS-sGUtLfWt0a8lUNPZS_1YY55AqoiyrTiaDA4bI11ntCQMNePqykHVeyFwUN1Gpola1SOG4mT-JtA9jmFbv3ulZizPTcXwAGWtLow:1thVJC:MKVvh0Wss3C2Gzn-B7dXTEyWjmhPDlpD_ZbIZb1fz58', '2025-02-24 15:02:50.955094');
INSERT INTO `django_session` VALUES ('7fy2njqna784v1wsg2qa2g08ack8uddg', '.eJyFj0mKwzAQRa_S1NoCSbai4TKiNLVF3HZwyQQ65O4thyyy6-X7A1X_AZX8sn3XFVzbjzzAC_xBeQf3gJrAiQFOXPEng4OWqcEAa43Xt9LphkT3bU9-Rpq7pKMwtlyCuFhRUhhLtjpFXXKacrAyflYIl9YrYhwnI7tBDdtBr7Nxz9iyx9OXXCrGJRP8S0yOK6dEDx-39E-i78Ojzdtef_uzBRfKzwEa0tWf40BbjiFKw7hWkU1WGxaCMKwoqYXCNHIb4PkH2hhYiQ:1tjf5K:iElO7quBgrkOFYXHeF3ZkdNYE3ttyS3NH7PT4lh5GUo', '2025-03-02 13:53:26.152781');
INSERT INTO `django_session` VALUES ('f3lmoqdap0tktwqk1khjxi62qk87qewa', '.eJyFj8sKgzAQRX-lzFrBxHd-RsZkUkOtSiahUPHfG6WL7ro898HM3cHxMK93t4AKPlIGFwyRyYPawRlQIoMTF3wSKAjEATJYnH58lUQbMr9Wb4YJeUpSq0XX22YUTS-sGUtLfWt0a8lUNPZS_1YY55AqoiyrTiaDA4bI11ntCQMNePqykHVeyFwUN1Gpola1SOG4mT-JtA9jmFbv3ulZizPTcXwAGWtLow:1thVmL:isk21HkHSaE5D2bMyIsj91rloPI788CcVqCCXsqiD88', '2025-02-24 15:32:57.537859');
INSERT INTO `django_session` VALUES ('ixfwzj2wm80d7o5hvnmruxl6lpdhzofx', '.eJyFj01uxCAMha9SeR0kIIQkXAYZME00aTLCRCN1NHcvGXXRXZff-5H9nrCy347PdQdXy0kdvMGfTAXcE9YETnVw4Y5fBA4qcYUO9jXefpVGd2R-HCX5BXlp0hjVNGcblJ1VTqHPNI8pjpmSoTDr-LfCuNVWUX1vJt0MrlhPfp-NhbCSx8vXUg9CaqHkhzJODm5QLXze0z-Jtg_Puhxl_W7PZtyYXh1U5Ju_xoE2RirUJFRvjTB5SmLqgxUBEyY52hhGhNcP2BlZBA:1thXtb:YSUwj-Tjl4a23_3OvxpNextYHv4woBm9JeeU_fDjVXk', '2025-02-24 17:48:35.742411');
INSERT INTO `django_session` VALUES ('lsen51aroruz4awy0e50verwt4d9ojwy', '.eJyFj8sKgzAQRX-lzFrBxHd-RsZkUkOtSiahUPHfG6WL7ro898HM3cHxMK93t4AKPlIGFwyRyYPawRlQIoMTF3wSKAjEATJYnH58lUQbMr9Wb4YJeUpSq0XX22YUTS-sGUtLfWt0a8lUNPZS_1YY55AqoiyrTiaDA4bI11ntCQMNePqykHVeyFwUN1Gpola1SOG4mT-JtA9jmFbv3ulZizPTcXwAGWtLow:1thV9F:LcBhNQnRdCZRtsLUcXUQ32Nut5RtELLi6dcOt7s4aUM', '2025-02-24 14:52:33.670921');
INSERT INTO `django_session` VALUES ('mpc3cjsd9fudsua1ct7ww5h11dj77v2t', '.eJyFj01uxCAMha9SeR0kSCAELhM5YBo0aTLCRCN1NHcvGXXRXZff-5H9npB53o7PvIOv5aQO3jCfTAX8E3IErzq4cMcvAg-VuEIHew63X6XRHZkfR4nzirw2yQY1uTQuanQqxWVI5GwMNlHUtLg-_K0wbrVV1DDoqW8GV6wnv8-GQlhpxsvvZW-E7IWSH0p7abxRLXze4z-Jtg_Puh4lf7dnE25Mrw4q8m2-xkGy1hgTtQh6IqGNXASSSsIiaSmdCyMivH4A4jdZGQ:1ti771:J54SA6cfuGKcs61qAl-_0ld4-aBgUFaRFvdORajfGmU', '2025-02-26 07:24:47.513418');
INSERT INTO `django_session` VALUES ('ptzsgsquth6p1jlei606j63qt4nvq31e', '.eJyFj8sKgzAQRX-lzFrBxHd-RsZkUkOtSiahUPHfG6WL7ro898HM3cHxMK93t4AKPlIGFwyRyYPawRlQIoMTF3wSKAjEATJYnH58lUQbMr9Wb4YJeUpSq0XX22YUTS-sGUtLfWt0a8lUNPZS_1YY55AqoiyrTiaDA4bI11ntCQMNePqykHVeyFwUN1Gpola1SOG4mT-JtA9jmFbv3ulZizPTcXwAGWtLow:1thVRt:uiWRNLikcXYyStv8jHKYNfRStRB0nYK8M9soaNSY6dM', '2025-02-24 15:11:49.725013');
INSERT INTO `django_session` VALUES ('rmhfdhfeasvu0b7hk3wqv0tznlntob0r', '.eJyFj01uxCAMha9SeR0kSEgIXCYyYBo0aTLCRCN1NHcvGXXRXZff-5H9npB52Y7PvIOr5aQO3rCcTAXcE3IEpzq4cMcvAgeVuEIHew63X6XRHZkfR4nLirw2yQQ12zR5NVmVoh8SWRODSRQ1eduHvxXGrbaKGgY9983givXk99lQCCstePm97Eche6Hkh9JOjm5ULXze4z-Jtg_Puh4lf7dnE25Mrw4q8m25xoElr42kQVg7aaF9lAIniyLNEr20hvSs4PUD4ElYwQ:1thXko:ws4JcUrzMTSAi6RWJqFgxyOGbvv3dRCmltArTP3HlQc', '2025-02-24 17:39:30.961731');
INSERT INTO `django_session` VALUES ('xoobqvi44jvcqqrg7guczed64v2c08mi', '.eJyFj8sKgzAQRX-lzFrBxHd-RsZkUkOtSiahUPHfG6WL7ro898HM3cHxMK93t4AKPlIGFwyRyYPawRlQIoMTF3wSKAjEATJYnH58lUQbMr9Wb4YJeUpSq0XX22YUTS-sGUtLfWt0a8lUNPZS_1YY55AqoiyrTiaDA4bI11ntCQMNePqykHVeyFwUN1Gpola1SOG4mT-JtA9jmFbv3ulZizPTcXwAGWtLow:1thWIr:INAtGEUKkhj3uaCJznL9YsGII-mvPhR9LyjWTKYPKpc', '2025-02-24 16:06:33.565578');
INSERT INTO `django_session` VALUES ('zamssiol77smgwhl1vauilzctz3qy549', '.eJyFT8tugzAQ_JVqz1jyAwfsn0Fre11QKEReo0iN8u81UQ659TgvzcwDFp7W_XvZwNdyUAcvMB1MBfwDlgRedXDCDX8IPFTiCh1sS7y-mYZuyHzfS5pm5LlRQ1Sjy5egLk7lFEwmN6Q4ZEo9BafjZ4RxrS2ijOlH3QSuWA9-1cZCWGnCU9dSWyG1UPJL9V5ab1UzH7f0j6P9w6POe1l-29iMK9Ozg4p8nc5zkElpY-UoZAhG9Mlq4TANIrW95CTmbEd4_gHjiVlH:1thUuu:_Kb4J3hWwmw9U1WhNfKL1sOutYtMT2YC4zgkYBC2-20', '2025-02-24 14:37:44.639255');

-- ----------------------------
-- Table structure for main_ipaddressrule
-- ----------------------------
DROP TABLE IF EXISTS `main_ipaddressrule`;
CREATE TABLE `main_ipaddressrule`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `ip_address` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `rule_type` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `description` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `created_at` datetime(6) NOT NULL,
  `updated_at` datetime(6) NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `ip_address`(`ip_address`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of main_ipaddressrule
-- ----------------------------

-- ----------------------------
-- Table structure for records
-- ----------------------------
DROP TABLE IF EXISTS `records`;
CREATE TABLE `records`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `task_id` char(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `temp_result_file_path` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `start_time` datetime(6) NOT NULL,
  `end_time` datetime(6) NULL DEFAULT NULL,
  `exec_time` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `status` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `user_id` bigint NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `task_id`(`task_id`) USING BTREE,
  INDEX `records_user_id_5b0abfdf_fk_user_id`(`user_id`) USING BTREE,
  CONSTRAINT `records_user_id_5b0abfdf_fk_user_id` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 44 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of records
-- ----------------------------
INSERT INTO `records` VALUES (1, 'fe1235080bb34d529ad7ddc7e90aff58', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmp94zml2nw', '2025-02-10 14:37:44.625343', '2025-02-10 14:37:47.496460', '0', 'completed', 1);
INSERT INTO `records` VALUES (2, 'b62071e7432e45d4b01707f645790035', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpqniq3kw9', '2025-02-10 17:10:38.981878', '2025-02-10 17:10:41.422589', '0', 'completed', 1);
INSERT INTO `records` VALUES (3, 'b731da16e5ae40168f9fda48cd20ad96', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpk7dsx9u3', '2025-02-10 17:16:48.419662', '2025-02-10 17:16:48.420659', NULL, 'pending', 1);
INSERT INTO `records` VALUES (4, 'a3b608e71db34c4bb0b8dee5ee48c6d7', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpypkpmr24', '2025-02-10 17:18:16.653881', '2025-02-10 17:18:16.654878', NULL, 'pending', 1);
INSERT INTO `records` VALUES (5, 'ba7be0e3208148be8f22470f884aa5f2', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpjy__k9p0', '2025-02-10 17:19:30.006556', '2025-02-10 17:19:42.029091', '0', 'completed', 1);
INSERT INTO `records` VALUES (6, '1a44c18d01d7480080ecfa9851b66581', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmp3camybtq', '2025-02-10 17:28:53.447409', '2025-02-10 17:28:55.423231', '0', 'completed', 1);
INSERT INTO `records` VALUES (7, 'd097953b03ad45fdafd4063dc34ed082', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpgrd75brw', '2025-02-10 17:29:37.028581', '2025-02-10 17:29:39.232980', '0', 'completed', 1);
INSERT INTO `records` VALUES (8, '8a8cf9bbfdfb4ea2a0b986e5368c2b14', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmp2ln5gv9d', '2025-02-10 17:29:46.588482', '2025-02-10 17:29:48.563677', '0', 'completed', 1);
INSERT INTO `records` VALUES (9, '7e8f419df3104d188cc4ef058d624dab', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpnwdf_419', '2025-02-10 17:31:50.357670', '2025-02-10 17:31:53.955885', '0', 'completed', 1);
INSERT INTO `records` VALUES (10, 'a2413f3c37874db995cc18449b6949f4', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpmer71pz3', '2025-02-10 17:32:13.773610', '2025-02-10 17:32:15.860706', '0', 'completed', 1);
INSERT INTO `records` VALUES (11, 'edf281d91d434726831d5e7e16eda8d3', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmp0sgnpz_r', '2025-02-10 17:32:21.599237', '2025-02-10 17:32:23.813408', '0', 'completed', 1);
INSERT INTO `records` VALUES (12, '1bc2c51a6e10498e9b7c69bceec16757', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpcxu1nhww', '2025-02-10 17:33:15.736604', '2025-02-10 17:33:17.827572', '0', 'completed', 1);
INSERT INTO `records` VALUES (13, '4a3589dc03a44b9489359f8ffeb37b6d', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmp6upile8g', '2025-02-10 17:34:28.441409', '2025-02-10 17:34:30.603141', '0', 'completed', 1);
INSERT INTO `records` VALUES (14, '6d5384bce7b44fc29da647fc7706c799', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpnbl7mer0', '2025-02-10 17:34:32.507836', '2025-02-10 17:34:34.715836', '0', 'completed', 1);
INSERT INTO `records` VALUES (15, '00b313891b3c4002a4e13b259ed6f25f', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpisru8vo0', '2025-02-10 17:35:18.374610', '2025-02-10 17:35:20.490614', '0', 'completed', 1);
INSERT INTO `records` VALUES (16, '6a10233250b44b8fa267b5c1ca79b2c2', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmperh33rw3', '2025-02-10 17:39:16.254890', '2025-02-10 17:39:20.936229', '0', 'completed', 1);
INSERT INTO `records` VALUES (17, '9eb470e399644bd0a69af80ab097e481', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpsnzbks3p', '2025-02-10 17:39:30.956744', '2025-02-10 17:39:33.112020', '0', 'completed', 1);
INSERT INTO `records` VALUES (18, '0acdba3cf879497ca909bd7d57329981', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmps3ie26z3', '2025-02-10 17:48:22.478660', '2025-02-10 17:48:24.666698', '0', 'completed', 1);
INSERT INTO `records` VALUES (19, '8c6e89f426d4486a8c180169408d7137', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmp8q10ccbg', '2025-02-10 17:48:32.271072', '2025-02-10 17:48:34.382989', '0', 'completed', 1);
INSERT INTO `records` VALUES (20, '24401a2e13644f8d83b6badad076cb7a', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmp_jvar1wk', '2025-02-10 17:48:35.738422', '2025-02-10 17:48:37.779497', '0', 'completed', 1);
INSERT INTO `records` VALUES (21, '92216116c67146aaac881fad6e5d4ff6', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmp_7t2nlbw', '2025-02-10 17:49:47.094759', '2025-02-10 17:49:49.606590', '0', 'completed', 1);
INSERT INTO `records` VALUES (22, 'a3cbb4450452478d90c20b2a0e092181', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpmufco0nm', '2025-02-11 18:15:16.779106', '2025-02-11 18:15:16.779106', NULL, 'pending', 1);
INSERT INTO `records` VALUES (23, '40019d39aeb844a5b6c7245a87b07fe4', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpns4p_1sr', '2025-02-11 18:15:39.553031', '2025-02-11 18:15:53.927104', '0', 'completed', 1);
INSERT INTO `records` VALUES (24, 'e75e762432e141b78053673158f66361', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpbh2hsnei', '2025-02-11 18:31:15.125176', '2025-02-11 18:31:35.523064', '0', 'completed', 1);
INSERT INTO `records` VALUES (25, '20a355e38f6b4b0f9a57ca50b1f01b22', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpd5pqc29w', '2025-02-11 18:32:14.151305', '2025-02-11 18:32:14.151305', NULL, 'pending', 1);
INSERT INTO `records` VALUES (26, 'a5a901b32fb847fdb3b946ac4728a07d', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpc7th0xim', '2025-02-11 18:33:42.517918', '2025-02-11 18:33:47.437760', '0', 'completed', 1);
INSERT INTO `records` VALUES (27, '8682337a09504c1da81c3d155c27dfdd', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpxhmzlum_', '2025-02-11 18:33:50.455031', '2025-02-11 18:33:50.455031', NULL, 'pending', 1);
INSERT INTO `records` VALUES (28, '0ff22ea7f8e94b6092a07f8778bbad46', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmprfqtnt4w', '2025-02-11 18:41:00.020373', '2025-02-11 18:41:00.021405', NULL, 'pending', 1);
INSERT INTO `records` VALUES (29, '2b6a1faba55749ef99b574e651cd8e87', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmp5d4u64q7', '2025-02-11 18:41:13.902285', '2025-02-11 18:41:20.070675', '0', 'completed', 1);
INSERT INTO `records` VALUES (30, '95f08c3679fe4517a570c870750bce5a', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpnjxa4zcs', '2025-02-11 18:43:33.866920', '2025-02-11 18:43:39.551828', '0', 'completed', 1);
INSERT INTO `records` VALUES (31, '87598a7321aa42679a207089521298c7', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmp2sau2lh2', '2025-02-11 18:44:22.794629', '2025-02-11 18:44:26.485086', '0', 'completed', 1);
INSERT INTO `records` VALUES (32, 'e1afcb2eca5e4a20b8096e1b6fc78408', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpecm1eelj', '2025-02-11 18:44:35.965492', '2025-02-11 18:44:35.965492', NULL, 'pending', 1);
INSERT INTO `records` VALUES (33, 'e5bdb6c50ff0417ea95073384d79ec3c', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpdtvce392', '2025-02-11 19:06:25.418169', '2025-02-11 19:06:40.132510', '0', 'completed', 1);
INSERT INTO `records` VALUES (34, '55ef04f5024a44d89f931f73b5283561', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpis_k8k4r', '2025-02-11 19:06:30.279797', '2025-02-11 19:06:30.279797', NULL, 'pending', 1);
INSERT INTO `records` VALUES (35, 'b54684afb5c94f878f0d87e8c5f69966', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmp4bdx6ivp', '2025-02-11 19:07:06.328589', '2025-02-11 19:07:11.814583', '0', 'completed', 1);
INSERT INTO `records` VALUES (36, '9ad911d738f9408ca8546bea3fb563e1', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpl7sjzjy1', '2025-02-11 19:07:21.734317', '2025-02-11 19:07:33.651476', '0', 'completed', 1);
INSERT INTO `records` VALUES (37, '571c5fae4eb8410aa65856fba8a33d5f', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpwq46kuii', '2025-02-12 07:14:24.337397', '2025-02-12 07:15:06.110971', '0', 'completed', 1);
INSERT INTO `records` VALUES (38, '0ba3e431cabe4284b97eec644ddfebf5', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpr9iy7_lh', '2025-02-12 07:16:58.546329', '2025-02-12 07:17:09.236575', '0', 'completed', 1);
INSERT INTO `records` VALUES (39, 'f6b1019479004258a61213f9535ce94c', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpmafdj8v4', '2025-02-12 07:24:30.910088', '2025-02-12 07:24:40.995273', '0', 'completed', 1);
INSERT INTO `records` VALUES (40, 'f77555d4c48e450bae1f7ae40099c6aa', 'F:\\安全产品项目\\基于深度学习的入侵检测系统-过采样和蒸馏\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpy6l03trz', '2025-02-12 07:24:47.510390', '2025-02-12 07:24:57.953509', '0', 'completed', 1);
INSERT INTO `records` VALUES (41, '5fa5d84330c14f02a1718e9c17a85c30', 'F:\\安全产品项目\\基于深度学习的入侵检测系统新版UI\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmpg_g3a89h', '2025-02-16 13:31:59.506563', '2025-02-16 13:31:59.506563', NULL, 'pending', 1);
INSERT INTO `records` VALUES (42, '0dcbb075d8254d32bf443ee5fcd679e9', 'F:\\安全产品项目\\基于深度学习的入侵检测系统新版UI\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmp0b4bz68w', '2025-02-16 13:33:20.491509', '2025-02-16 13:33:33.392113', '0', 'completed', 1);
INSERT INTO `records` VALUES (43, '790abc28075c4978bb18f52715ad309b', 'F:\\安全产品项目\\基于深度学习的入侵检测系统新版UI\\g-qusz7812-dl_idsdl_ids-dl_ids-\\static\\tmp\\tmp55kitosp', '2025-02-16 13:53:26.147048', '2025-02-16 13:53:35.157770', '0', 'completed', 1);

-- ----------------------------
-- Table structure for tuning_models
-- ----------------------------
DROP TABLE IF EXISTS `tuning_models`;
CREATE TABLE `tuning_models`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `tuning_id` char(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `tuning_model` varchar(64) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `start_time` datetime(6) NOT NULL,
  `end_time` datetime(6) NULL DEFAULT NULL,
  `lr` double NOT NULL,
  `wd` double NOT NULL,
  `batch_size` int NOT NULL,
  `num_epochs` int NOT NULL,
  `alpha` int NOT NULL,
  `temperature` int NOT NULL,
  `accuracy` double NULL DEFAULT NULL,
  `loss` double NULL DEFAULT NULL,
  `test_accuracy` double NULL DEFAULT NULL,
  `created_at` datetime(6) NOT NULL,
  `user_id` bigint NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `tuning_id`(`tuning_id`) USING BTREE,
  INDEX `tuning_models_user_id_60025de8_fk_user_id`(`user_id`) USING BTREE,
  CONSTRAINT `tuning_models_user_id_60025de8_fk_user_id` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 19 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of tuning_models
-- ----------------------------
INSERT INTO `tuning_models` VALUES (1, '5c46420d0b5140f38027d6271fed7377', 'LSTM', '2025-02-10 14:36:13.826012', '2025-02-10 14:36:13.884855', 0.000005, 0.000006, 256, 20, 0, 2, 90.64, 50.68, 74.16, '2025-02-10 14:36:13.885854', 1);
INSERT INTO `tuning_models` VALUES (2, '2f893e46f0444cc082ea9945f4d68c2f', 'LSTM', '2025-02-10 14:37:18.538889', '2025-02-10 14:37:18.538889', 0.000005, 0.000006, 256, 10, 0, 1, 68.95, 59.36, 64.59, '2025-02-10 14:37:18.539886', 1);
INSERT INTO `tuning_models` VALUES (3, '3775ce07c3a441c6bef22641bcdbab7f', 'LSTM', '2025-02-10 14:42:55.530572', '2025-02-10 14:42:55.530572', 0.000005, 0.000006, 256, 10, 0, 2, 87.98, 58, 81.52, '2025-02-10 14:42:55.530572', 1);
INSERT INTO `tuning_models` VALUES (4, '4fd5ba59e5e449ab91345e53c4fb41b3', 'LSTM multiple', '2025-02-10 14:52:50.027942', '2025-02-10 14:52:50.028940', 0.000005, 0.000006, 256, 10, 0, 2, 0, 3.64, 0.3, '2025-02-10 14:52:50.029938', 1);
INSERT INTO `tuning_models` VALUES (5, 'b9a43f5ae7a74c07a9fd9fc8c3b25b68', 'LSTM multiple', '2025-02-10 14:59:43.626372', '2025-02-10 14:59:43.626372', 0.000009, 0.000006, 256, 20, 0, 2, 1300, 3.64, 15.57, '2025-02-10 14:59:43.627369', 1);
INSERT INTO `tuning_models` VALUES (6, 'b1fae8f05c5f460dabe395adb69da9c1', 'LSTM multiple', '2025-02-10 15:03:56.802462', '2025-02-10 15:03:56.803459', 0.000001, 0.000006, 256, 20, 0, 2, 100, 3.64, 0.02, '2025-02-10 15:03:56.804457', 1);
INSERT INTO `tuning_models` VALUES (7, '993b17b3e93a4ad585826784982e4ee9', 'LSTM multiple', '2025-02-10 15:06:41.886494', '2025-02-10 15:06:41.887491', 0.000001, 0.000006, 256, 20, 0, 2, 100, 3.64, 0.67, '2025-02-10 15:06:41.888489', 1);
INSERT INTO `tuning_models` VALUES (8, '33ac176599bd4c8084814f23da1ae370', 'LSTM multiple', '2025-02-10 15:07:42.750072', '2025-02-10 15:07:42.750072', 0.000001, 0.000006, 256, 20, 0, 2, 0, 3.64, 1.03, '2025-02-10 15:07:42.751069', 1);
INSERT INTO `tuning_models` VALUES (9, '2fdd07072f48427099751f82ef1f9739', 'LSTM multiple', '2025-02-10 15:07:48.781068', '2025-02-10 15:07:48.781068', 0.000001, 0.000006, 256, 20, 0, 2, 700, 3.64, 11.5, '2025-02-10 15:07:48.781068', 1);
INSERT INTO `tuning_models` VALUES (10, '56632b9e737348fe822298bae97af052', 'LSTM multiple', '2025-02-10 15:09:08.875076', '2025-02-10 15:09:08.876073', 0.000008, 0.000006, 256, 20, 0, 2, 900, 3.64, 10.69, '2025-02-10 15:09:08.877071', 1);
INSERT INTO `tuning_models` VALUES (11, 'b78e8c4e3dd84f7d96392d3eae04965d', 'LSTM multiple', '2025-02-10 15:14:39.231382', '2025-02-10 15:14:39.232380', 0.000081, 0.000006, 256, 20, 0, 2, 9700, 2.79, 66.77, '2025-02-10 15:14:39.233377', 1);
INSERT INTO `tuning_models` VALUES (12, 'a28d8ea5129d4f8faf276b66fd0c5c71', 'LSTM multiple', '2025-02-10 15:35:40.278757', '2025-02-10 15:35:40.279756', 0.000081, 0.000006, 256, 20, 0, 2, 95.77, 271.03, 67.34, '2025-02-10 15:35:40.280753', 1);
INSERT INTO `tuning_models` VALUES (13, 'e2b3434b970c4d12a3825189a24cf268', 'LSTM', '2025-02-11 18:15:06.527684', '2025-02-11 18:15:06.528685', 0.000081, 0.000006, 256, 20, 0, 2, 98.76, 0.03, 80.69, '2025-02-11 18:15:06.528685', 1);
INSERT INTO `tuning_models` VALUES (14, '847f2128476045d78676ae25402860c2', 'LSTM multiple', '2025-02-11 18:19:41.734336', '2025-02-11 18:19:41.734336', 0.000081, 0.000006, 256, 20, 0, 2, 32.87, 2.81, 20.65, '2025-02-11 18:19:41.734336', 1);
INSERT INTO `tuning_models` VALUES (15, 'cef6158787d0427e8299700617712370', 'LSTM multiple', '2025-02-12 07:21:15.757271', '2025-02-12 07:21:15.757271', 0.000508, 0.000006, 256, 10, 0, 2, 90.62, 2.71, 63.36, '2025-02-12 07:21:15.757271', 1);
INSERT INTO `tuning_models` VALUES (16, '3dd8ce5fca2745b7aaf750001db8329f', 'LSTM', '2025-02-16 13:31:20.872609', '2025-02-16 13:31:20.889677', 0.000081, 0.000006, 256, 20, 0, 2, 99.6, 0.03, 76.97, '2025-02-16 13:31:20.890674', 1);
INSERT INTO `tuning_models` VALUES (17, '0cc3b352e6084e56af8e309f6f3d878d', 'LSTM multiple', '2025-02-16 13:42:02.028112', '2025-02-16 13:42:02.028112', 0.000508, 0.000006, 256, 10, 0, 2, 62.24, 2.74, 56.98, '2025-02-16 13:42:02.028112', 1);
INSERT INTO `tuning_models` VALUES (18, '1e88448f3f704e7a9d58aebbb62e4755', 'LSTM', '2025-02-16 13:45:46.834367', '2025-02-16 13:45:46.834367', 0.000081, 0.000006, 256, 20, 0, 2, 95.28, 0.07, 86.64, '2025-02-16 13:45:46.835364', 1);

-- ----------------------------
-- Table structure for user
-- ----------------------------
DROP TABLE IF EXISTS `user`;
CREATE TABLE `user`  (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `username` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `nickname` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `password_hash` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `password_salt` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `status` int NOT NULL,
  `create_at` datetime(6) NOT NULL,
  `update_at` datetime(6) NOT NULL,
  `is_authorize` tinyint(1) NOT NULL,
  `is_change_file_type` tinyint(1) NOT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 2 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of user
-- ----------------------------
INSERT INTO `user` VALUES (1, 'test', '', '7c189f6b1691fdb3fe97dc7fed4eb92c', '133482', 1, '2025-02-10 14:05:51.000000', '2025-02-10 14:05:51.000000', 0, 0);

SET FOREIGN_KEY_CHECKS = 1;
