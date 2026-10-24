-- Run in your own Supabase project. Only derived profiles and reviewed menus are saved.
create table if not exists public.palatable_profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  profile jsonb,
  menus jsonb not null default '[]'::jsonb,
  updated_at timestamptz not null default now()
);
alter table public.palatable_profiles enable row level security;
create policy "Read own taste profile" on public.palatable_profiles for select to authenticated using ((select auth.uid()) = user_id);
create policy "Create own taste profile" on public.palatable_profiles for insert to authenticated with check ((select auth.uid()) = user_id);
create policy "Update own taste profile" on public.palatable_profiles for update to authenticated using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id);
create policy "Delete own taste profile" on public.palatable_profiles for delete to authenticated using ((select auth.uid()) = user_id);
grant select, insert, update, delete on public.palatable_profiles to authenticated;
