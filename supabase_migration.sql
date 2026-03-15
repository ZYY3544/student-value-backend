-- ============================================
-- Supabase 数据表迁移脚本
-- 在 Supabase Dashboard → SQL Editor 中执行
-- ============================================

-- 用户档案（auth.users 由 Supabase Auth 自动管理）
CREATE TABLE IF NOT EXISTS user_profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  display_name TEXT,
  school_name TEXT,
  education_level TEXT,
  major TEXT,
  preferred_city TEXT,
  preferred_industry TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 评估记录
CREATE TABLE IF NOT EXISTS assessments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  resume_text TEXT,
  form_data JSONB,
  result JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 对话会话
CREATE TABLE IF NOT EXISTS chat_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  assessment_id UUID REFERENCES assessments(id) ON DELETE CASCADE,
  phase TEXT DEFAULT 'opening',
  conversation_memory JSONB,
  pinned BOOLEAN DEFAULT FALSE,
  title TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 补充列（已有表则追加）
ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS pinned BOOLEAN DEFAULT FALSE;
ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS title TEXT;
ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS assessment_context JSONB;
ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS resume_text TEXT;

-- 对话消息
CREATE TABLE IF NOT EXISTS chat_messages (
  id BIGSERIAL PRIMARY KEY,
  session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- RLS 策略：用户只能访问自己的数据
-- ============================================

ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;

-- user_profiles: 用户只能操作自己的资料
CREATE POLICY "users_own_profile" ON user_profiles
  FOR ALL USING (auth.uid() = id);

-- assessments: 用户只能访问自己的评估
CREATE POLICY "users_own_assessments" ON assessments
  FOR ALL USING (auth.uid() = user_id);

-- chat_sessions: 用户只能访问自己的会话
CREATE POLICY "users_own_sessions" ON chat_sessions
  FOR ALL USING (auth.uid() = user_id);

-- chat_messages: 用户只能访问自己会话中的消息
CREATE POLICY "users_own_messages" ON chat_messages
  FOR ALL USING (
    session_id IN (SELECT id FROM chat_sessions WHERE user_id = auth.uid())
  );

-- ============================================
-- 索引优化
-- ============================================

CREATE INDEX IF NOT EXISTS idx_assessments_user_id ON assessments(user_id);
CREATE INDEX IF NOT EXISTS idx_assessments_created_at ON assessments(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);
